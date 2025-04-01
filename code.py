pip install tf-keras-vis

import kagglehub
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path = kagglehub.dataset_download("esfiam/brain-tumor-mri-dataset")

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_gen = ImageDataGenerator(
    rescale=1./255
)

train = train_gen.flow_from_directory(
    '/root/.cache/kagglehub/datasets/esfiam/brain-tumor-mri-dataset/versions/1/train',
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test = test_gen.flow_from_directory(
    '/root/.cache/kagglehub/datasets/esfiam/brain-tumor-mri-dataset/versions/1/test',
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Input(shape=(256, 256, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='brain_tumor_mri.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3, min_lr=1e-5)
early_stop = EarlyStopping(monitor='loss', patience=5)

history = model.fit(train,
                    epochs=100,
                    validation_data=test,
                    callbacks=[checkpoint, reduce_lr, early_stop]
                   )

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

loss, accuracy = model.evaluate(test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

import os
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

test_dir = '/root/.cache/kagglehub/datasets/esfiam/brain-tumor-mri-dataset/versions/1/test'
classes = list(test.class_indices.keys())
test_images, true_labels = [], []


for class_name in classes:
    class_dir = os.path.join(test_dir, class_name)
    all_images = os.listdir(class_dir)
    selected_images = random.sample(all_images, 5)

    for img_name in selected_images:
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path, target_size=(256, 256), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)
        true_labels.append(class_name)


test_images = np.array(test_images)


predictions = model.predict(test_images)
predicted_labels = [classes[np.argmax(pred)] for pred in predictions]


plt.figure(figsize=(20, 20))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.axis('off')


    color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
    plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", color=color)

plt.tight_layout()
plt.show()

import numpy as np

# Get all test images and labels
test.reset()  # Important to avoid skipping samples
predictions = model.predict(test, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test.classes
class_labels = list(test.class_indices.keys())
from sklearn.metrics import classification_report

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:\n", report)

import cv2
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import cm

# Utility function: convert grayscale image to 3-channel RGB.
def to_rgb(img):
    return np.repeat(img, 3, axis=-1)

# Define a color map for each class index.
# Colors: 0: Red, 1: Green, 2: Blue, 3: Yellow.
color_map = {
    0: np.array([1, 0, 0]),   # Red
    1: np.array([0, 1, 0]),   # Green
    2: np.array([0, 0, 1]),   # Blue
    3: np.array([1, 1, 0])    # Yellow
}

# Create a GradCAM instance for your trained model.
gradcam = Gradcam(model, model_modifier=None, clone=True)

def advanced_segmentation(image, pred_class, gradcam_instance, color_map, threshold=0.5, min_area=100):
    """
    Generate an advanced segmentation overlay using Grad-CAM.

    Parameters:
      image         : Input image (256x256x1, normalized)
      pred_class    : Predicted class index for the image
      gradcam_instance: A Gradcam object for the model
      color_map     : Dictionary mapping class index to an RGB color (values in [0,1])
      threshold     : Threshold for converting the heatmap to a binary mask
      min_area      : Minimum contour area to consider for drawing bounding boxes

    Returns:
      cam           : The computed Grad-CAM heatmap (normalized)
      binary_mask   : The binary mask obtained from thresholding the heatmap
      overlay       : Original image overlaid with the colored heatmap and bounding boxes
    """
    # Compute Grad-CAM heatmap.
    input_img = np.expand_dims(image, axis=0)
    score = CategoricalScore(pred_class)
    cam = gradcam_instance(score, input_img)[0]

    # Normalize the heatmap.
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Create a colored heatmap using the jet colormap.
    colored_heatmap = cm.jet(cam)[..., :3]  # shape (H, W, 3) with values in [0,1]
    colored_heatmap = np.float32(colored_heatmap)

    # Generate a binary mask from the Grad-CAM heatmap.
    binary_mask = np.uint8((cam > threshold) * 255)

    # Use morphological operations to clean up the mask.
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the binary mask.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the original grayscale image to RGB.
    original_rgb = (to_rgb(image) * 255).astype(np.uint8)
    overlay = original_rgb.copy()

    # Draw bounding boxes for contours with area greater than min_area.
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            box_color = (color_map[pred_class] * 255).astype(int).tolist()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), box_color, 2)

    # Blend the overlay with the colored heatmap.
    blended = cv2.addWeighted(overlay, 0.6, (colored_heatmap * 255).astype(np.uint8), 0.4, 0)

    return cam, binary_mask, blended

# Number of images per class.
num_per_class = 5
classes = list(test.class_indices.keys())  # e.g., ['glioma', 'meningioma', 'no_tumor', 'pituitary']
test_dir = '/root/.cache/kagglehub/datasets/esfiam/brain-tumor-mri-dataset/versions/1/test'
num_classes = len(classes)

# Create a figure: For each class, we'll have 3 rows (Original, Grad-CAM, Segmentation)
# Total rows = num_classes * 3; columns = num_per_class.
fig, axes = plt.subplots(nrows=num_classes * 3, ncols=num_per_class, figsize=(5 * num_per_class, 4 * num_classes * 3))
fig.suptitle("Advanced Segmentation Results: 5 Images per Class", fontsize=20, y=0.92)

# Loop through each class.
for cls_idx, class_name in enumerate(classes):
    class_dir = os.path.join(test_dir, class_name)
    all_images = os.listdir(class_dir)
    selected_images = random.sample(all_images, num_per_class)

    # For each image in this class.
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path, target_size=(256, 256), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0  # normalized image

        # Predict the class.
        input_img = np.expand_dims(img_array, axis=0)
        pred_class = np.argmax(model.predict(input_img))

        # Get advanced segmentation outputs.
        cam, binary_mask, overlay = advanced_segmentation(img_array, pred_class, gradcam, color_map, threshold=0.5, min_area=100)

        # Calculate row offsets.
        row_orig = cls_idx * 3      # Original image row for this class.
        row_cam  = cls_idx * 3 + 1   # Grad-CAM row.
        row_seg  = cls_idx * 3 + 2   # Segmentation overlay row.

        # Plot Original Image with a label.
        ax_orig = axes[row_orig][i] if num_per_class > 1 else axes[row_orig]
        ax_orig.imshow(to_rgb(img_array))
        if i == 0:
            ax_orig.set_ylabel(f"{class_name.upper()}\nOriginal", fontsize=14, rotation=0, labelpad=70)
        ax_orig.set_title(f"Image: {img_name}", fontsize=10)
        ax_orig.axis('off')

        # Plot Grad-CAM Heatmap.
        ax_cam = axes[row_cam][i] if num_per_class > 1 else axes[row_cam]
        ax_cam.imshow(cam, cmap='jet')
        if i == 0:
            ax_cam.set_ylabel("Grad-CAM", fontsize=14, rotation=0, labelpad=70)
        ax_cam.set_title(f"Pred: {classes[pred_class]}", fontsize=10)
        ax_cam.axis('off')

        # Plot Segmentation Overlay.
        ax_seg = axes[row_seg][i] if num_per_class > 1 else axes[row_seg]
        ax_seg.imshow(overlay)
        if i == 0:
            ax_seg.set_ylabel("Segmentation", fontsize=14, rotation=0, labelpad=70)
        ax_seg.set_title(f"Pred: {classes[pred_class]}", fontsize=10)
        ax_seg.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()


