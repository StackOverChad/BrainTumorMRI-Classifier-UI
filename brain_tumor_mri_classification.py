import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_v2_preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from IPython.display import clear_output
import glob

# --- 1. Download and Setup Dataset Path ---

import kagglehub
dataset_path_object = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("Path to dataset files:", dataset_path_object)
dataset_path_object = "/kaggle/input/brain-tumor-mri-dataset"
print("Path to dataset files:", dataset_path_object)

train_dir_base = os.path.join(dataset_path_object, 'Training')
test_dir_base = os.path.join(dataset_path_object, 'Testing')

if not os.path.exists(train_dir_base) or not os.path.exists(test_dir_base):
    print(f"WARNING: Training or Testing directory not found directly under {dataset_path_object}")
    dataset_slug = os.path.basename(dataset_path_object)
    if "brain-tumor-mri-dataset" not in dataset_slug:
        dataset_slug = "brain-tumor-mri-dataset"

    train_dir_nested = os.path.join(dataset_path_object, dataset_slug, 'Training')
    test_dir_nested = os.path.join(dataset_path_object, dataset_slug, 'Testing')

    if os.path.exists(train_dir_nested) and os.path.exists(test_dir_nested):
        print(f"Found nested training directory: {train_dir_nested}")
        train_dir = train_dir_nested
        test_dir = test_dir_nested
    elif os.path.exists(os.path.join(dataset_path_object, 'brain-tumor-mri-dataset', 'Training')):
        print("Found common nested structure: brain-tumor-mri-dataset/Training")
        train_dir = os.path.join(dataset_path_object, 'brain-tumor-mri-dataset', 'Training')
        test_dir = os.path.join(dataset_path_object, 'brain-tumor-mri-dataset', 'Testing')
    else:
        print("Could not find a known nested structure. Assuming direct Training/Testing folders.")
        train_dir = train_dir_base
        test_dir = test_dir_base
else:
    train_dir = train_dir_base
    test_dir = test_dir_base

print(f"Using Training directory: {train_dir}")
print(f"Using Testing directory: {test_dir}")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}. Please verify the dataset path and structure.")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Testing directory not found: {test_dir}. Please verify the dataset path and structure.")

print("\n--- Dataset File Counts ---")
for split_dir, split_name in [(train_dir, "Training"), (test_dir, "Testing")]:
    print(f"\n{split_name} Data:")
    if os.path.exists(split_dir):
        expected_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        actual_classes_in_split = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        print(f"Found class directories in {split_name}: {actual_classes_in_split}")

        for class_name in actual_classes_in_split:
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                num_files = len(glob.glob(os.path.join(class_path, '*.*')))
                print(f"Class '{class_name}': {num_files} images")
    else:
        print(f"Directory {split_dir} not found.")
print("---------------------------\n")


# ---  Define Parameters ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNE = 20
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

# --- Load and Preprocess Data ---
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print("Class names found by image_dataset_from_directory:", class_names)
print("Number of classes:", num_classes)
print(f"Number of training batches: {len(train_dataset)}")
print(f"Number of validation batches: {len(validation_dataset)}")
print(f"Number of test batches: {len(test_dataset)}")


# --- 4. Data Augmentation and Preprocessing Layers ---
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def preprocess_for_resnetv2(image, label):
    image = tf.cast(image, tf.float32)
    image = resnet_v2_preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

# Apply ResNetV2 preprocessing to all datasets
train_dataset = train_dataset.map(preprocess_for_resnetv2, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(preprocess_for_resnetv2, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset_for_eval = test_dataset.map(preprocess_for_resnetv2, num_parallel_calls=tf.data.AUTOTUNE)

# Configure datasets for performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset_for_eval = test_dataset_for_eval.prefetch(buffer_size=tf.data.AUTOTUNE)


# --- 5. Build the Model (Transfer Learning with ResNet50V2) ---
def build_model_resnet50v2(num_classes):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), name="dense_256")(x)
    x = layers.Dropout(0.5, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = keras.Model(inputs, outputs)
    return model

model = build_model_resnet50v2(num_classes)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_filepath = 'best_model_initial_resnet.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6)

# --- 8. Train the Model ---
print("\n--- Training Top Layers (ResNet50V2 based) ---")
history_initial = model.fit(
    train_dataset,
    epochs=EPOCHS_INITIAL,
    validation_data=validation_dataset,
    callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]
)
clear_output(wait=True)

if os.path.exists(checkpoint_filepath):
    model = tf.keras.models.load_model(checkpoint_filepath)
    print("Loaded best model from initial training phase checkpoint.")
else:
    print("No checkpoint found for initial training. Using the model as is after fitting.")


# --- 9. Fine-tuning ---
base_model_in_loaded_model = model.layers[1]
base_model_in_loaded_model.trainable = True

fine_tune_at_percentage = 0.70
fine_tune_at_index = int(len(base_model_in_loaded_model.layers) * fine_tune_at_percentage)
print(f"\nNumber of layers in the base model: {len(base_model_in_loaded_model.layers)}")
print(f"Will unfreeze base model layers from index: {fine_tune_at_index} ({base_model_in_loaded_model.layers[fine_tune_at_index].name})")

for layer in base_model_in_loaded_model.layers[:fine_tune_at_index]:
    layer.trainable = False
for layer in base_model_in_loaded_model.layers[fine_tune_at_index:]:
    if not isinstance(layer, layers.BatchNormalization):
         layer.trainable = True
    else:
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

checkpoint_filepath_ft = 'best_model_finetuned_resnet.keras'
model_checkpoint_callback_ft = ModelCheckpoint(
    filepath=checkpoint_filepath_ft,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback_ft = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True)

reduce_lr_callback_ft = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7)

print("\n--- Fine-tuning Model (ResNet50V2 based) ---")
initial_epochs_trained = len(history_initial.epoch) if hasattr(history_initial, 'epoch') and history_initial.epoch else 0

history_fine_tune = model.fit(
    train_dataset,
    epochs=initial_epochs_trained + EPOCHS_FINE_TUNE,
    initial_epoch=initial_epochs_trained,
    validation_data=validation_dataset,
    callbacks=[model_checkpoint_callback_ft, early_stopping_callback_ft, reduce_lr_callback_ft]
)
clear_output(wait=True)

# --- 10. Load the Best Overall Model and Evaluate on Test Data ---
best_overall_model = None
path_initial_best = checkpoint_filepath
path_finetuned_best = checkpoint_filepath_ft

val_acc_initial = max(history_initial.history.get('val_accuracy', [0])) if history_initial and history_initial.history else 0
val_acc_finetuned = max(history_fine_tune.history.get('val_accuracy', [0])) if history_fine_tune and history_fine_tune.history else 0

print(f"Best val_accuracy from initial training: {val_acc_initial:.4f}")
print(f"Best val_accuracy from fine-tuning: {val_acc_finetuned:.4f}")

if os.path.exists(path_finetuned_best) and val_acc_finetuned >= val_acc_initial :
    print(f"Loading best model from fine-tuning: {path_finetuned_best}")
    best_overall_model = tf.keras.models.load_model(path_finetuned_best)
elif os.path.exists(path_initial_best):
    print(f"Loading best model from initial training: {path_initial_best}")
    best_overall_model = tf.keras.models.load_model(path_initial_best)
else:
    print("WARNING: No saved model checkpoint found. Using current model state (likely from end of fine-tuning).")
    best_overall_model = model

if best_overall_model is None:
    print("ERROR: Could not load or determine the best model. Exiting evaluation.")
    exit()


print("\n--- Evaluating on Test Data (ResNet50V2 based) ---")
loss, accuracy = best_overall_model.evaluate(test_dataset_for_eval)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = best_overall_model.predict(test_dataset_for_eval)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

y_true = []
for _, labels_batch in test_dataset_for_eval:
    y_true.extend(np.argmax(labels_batch.numpy(), axis=1))
y_true = np.array(y_true)


# --- 11. Display All Metrics ---
print("\n--- Detailed Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4, zero_division=0))

print("\n--- Individual Metrics ---")
accuracy_val = accuracy_score(y_true, y_pred_classes)
precision_macro = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)

precision_weighted = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
recall_weighted = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
f1_weighted = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)

print(f"Accuracy (sklearn): {accuracy_val:.4f}")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall: {recall_weighted:.4f}")
print(f"Weighted F1-Score: {f1_weighted:.4f}")

precision_per_class = precision_score(y_true, y_pred_classes, average=None, zero_division=0)
recall_per_class = recall_score(y_true, y_pred_classes, average=None, zero_division=0)
f1_per_class = f1_score(y_true, y_pred_classes, average=None, zero_division=0)

print("\n--- Per-Class Metrics ---")
metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class
})
print(metrics_df)


# --- 12. Plot Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (ResNet50V2)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- 13. Plot Training History ---
def plot_history(history_initial, history_fine_tune=None, model_name="Model"):
    acc = history_initial.history.get('accuracy', [])
    val_acc = history_initial.history.get('val_accuracy', [])
    loss = history_initial.history.get('loss', [])
    val_loss = history_initial.history.get('val_loss', [])
    initial_epochs_range = range(len(acc))

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    if acc: plt.plot(initial_epochs_range, acc, label='Training Accuracy (Initial)')
    if val_acc: plt.plot(initial_epochs_range, val_acc, label='Validation Accuracy (Initial)')

    if history_fine_tune:
        acc_ft = history_fine_tune.history.get('accuracy', [])
        val_acc_ft = history_fine_tune.history.get('val_accuracy', [])
        initial_epochs_len = len(history_initial.epoch) if hasattr(history_initial, 'epoch') and history_initial.epoch else 0
        fine_tune_epochs_range = range(initial_epochs_len, initial_epochs_len + len(acc_ft))

        if acc_ft: plt.plot(fine_tune_epochs_range, acc_ft, label='Training Accuracy (Fine-tune)', linestyle='--')
        if val_acc_ft: plt.plot(fine_tune_epochs_range, val_acc_ft, label='Validation Accuracy (Fine-tune)', linestyle='--')

    plt.legend(loc='lower right')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)


    plt.subplot(1, 2, 2)
    if loss: plt.plot(initial_epochs_range, loss, label='Training Loss (Initial)')
    if val_loss: plt.plot(initial_epochs_range, val_loss, label='Validation Loss (Initial)')

    if history_fine_tune:
        loss_ft = history_fine_tune.history.get('loss', [])
        val_loss_ft = history_fine_tune.history.get('val_loss', [])
        initial_epochs_len = len(history_initial.epoch) if hasattr(history_initial, 'epoch') and history_initial.epoch else 0
        fine_tune_epochs_range = range(initial_epochs_len, initial_epochs_len + len(loss_ft))

        if loss_ft: plt.plot(fine_tune_epochs_range, loss_ft, label='Training Loss (Fine-tune)', linestyle='--')
        if val_loss_ft: plt.plot(fine_tune_epochs_range, val_loss_ft, label='Validation Loss (Fine-tune)', linestyle='--')

    plt.legend(loc='upper right')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if history_initial:
    plot_history(history_initial, history_fine_tune if 'history_fine_tune' in locals() and history_fine_tune else None, model_name="ResNet50V2")
else:
    print("Initial training history not available to plot.")


print("\n--- Model Training and Evaluation Complete (ResNet50V2 based) ---")

from google.colab import drive
import os

drive.mount('/content/drive')

project_drive_path = '/content/drive/My Drive/MyProjectFolder/BrainTumorSubfolder/'
os.makedirs(project_drive_path, exist_ok=True)

print(f"Google Drive mounted successfully.")
if os.path.exists(project_drive_path):
    print(f"Project path ready at: {project_drive_path}")
else:
    print(f"NOTE: The specific project path '{project_drive_path}' does not exist yet. You might need to create it manually in Drive, or uncomment os.makedirs to create it via code.")

# 1. Save the Keras model
model_save_path_colab = "brain_tumor_classifier_resnet50v2.keras"
best_overall_model.save(model_save_path_colab)
print(f"Model saved in Colab to: {model_save_path_colab}")

# 2. Save the class names
import json
class_names_colab = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_names_save_path_colab = "class_names.json"
with open(class_names_save_path_colab, "w") as f:
    json.dump(class_names_colab, f)
print(f"Class names saved in Colab to: {class_names_save_path_colab}")
print(f"Saved class names: {class_names_colab}")

# 3. Download these files to your local computer
from google.colab import files
files.download(model_save_path_colab)
files.download(class_names_save_path_colab)
print("Download prompts should appear for the model and class_names.json.")