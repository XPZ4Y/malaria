# python_assisted_modelTraining
*| Author: chris | Confidence: null | Topic: python_assisted_modelTraining #raw |*


## Context:
We are training a model to detect malaria from provided microscopic images. The final model weights ares stored, ready to be deployed. It is a simple A/B classification model.

It will be hosted in a backend, on node.

This is built to run on colab. To leverage foreign compute

## 1) Connecting to the google drive

```py
"""
Mount Google Drive Script for Colab
-----------------------------------
This script mounts Google Drive and sets up the working directory.
Run this first before downloading the dataset.
"""

import os
import sys

def mount_google_drive():
    """Mount Google Drive in Colab"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully at /content/drive")
        return True
    except ImportError:
        print("⚠ Not running in Google Colab. Skipping drive mount.")
        return False
    except Exception as e:
        print(f"✗ Error mounting drive: {e}")
        return False

def setup_working_directory(workspace_path="/content/malarai"):
    """Create and set up working directory"""
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
        print(f"✓ Created workspace directory: {workspace_path}")
    else:
        print(f"✓ Workspace directory exists: {workspace_path}")
    
    os.chdir(workspace_path)
    print(f"✓ Changed working directory to: {os.getcwd()}")
    return workspace_path

def check_directory_structure():
    """Check if dataset directory structure is correct"""
    if os.path.exists("dataset/Parasitized") and os.path.exists("dataset/Uninfected"):
        parasitized_count = len(os.listdir("dataset/Parasitized"))
        uninfected_count = len(os.listdir("dataset/Uninfected"))
        print(f"✓ Dataset structure verified!")
        print(f"  - Parasitized images: {parasitized_count}")
        print(f"  - Uninfected images: {uninfected_count}")
        return True
    else:
        print("⚠ Dataset directory structure not found. Please download the dataset first.")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Mounting Google Drive for Colab")
    print("=" * 50)
    
    # Mount Google Drive
    mount_google_drive()
    
    # Setup working directory
    setup_working_directory()
    
    # Optional: Copy dataset from Google Drive if already there
    drive_dataset_path = "/content/drive/MyDrive/malarai_dataset"
    if os.path.exists(drive_dataset_path):
        print(f"\n✓ Found dataset in Google Drive at {drive_dataset_path}")
        response = input("Do you want to copy it to the workspace? (y/n): ")
        if response.lower() == 'y':
            !cp -r {drive_dataset_path}/* /content/malarai/
            print("✓ Dataset copied from Google Drive")
            check_directory_structure()
    
    print("\n✓ Mount setup complete!")
```


## 2) Download script

```py
"""
Malaria Dataset Download Script for Colab (with Drive Persistence)
-------------------------------------------------------------------
Downloads and SAVES to Google Drive so you only download once.
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# 0. MOUNT GOOGLE DRIVE
# ─────────────────────────────────────────────

def mount_google_drive():
    """Mount Google Drive for persistent storage"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
        return True
    except ImportError:
        print("⚠ Not in Colab, using local storage only")
        return False
    except Exception as e:
        print(f"✗ Mount failed: {e}")
        return False

# ─────────────────────────────────────────────
# 1. DOWNLOAD TO DRIVE (PERSISTENT)
# ─────────────────────────────────────────────

def download_with_tensorflow_datasets():
    """Download dataset directly to Google Drive"""
    try:
        import tensorflow_datasets as tfds
        
        # Set persistent storage path on Google Drive
        DRIVE_DATASET_PATH = "/content/drive/MyDrive/malarai_dataset"
        LOCAL_CACHE = "/content/malarai_cache"  # Temporary cache
        
        print("Downloading malaria dataset from TensorFlow Datasets...")
        print("This may take 5-10 minutes depending on your connection...")
        
        # Download to local cache first (faster I/O)
        os.makedirs(LOCAL_CACHE, exist_ok=True)
        
        dataset, info = tfds.load(
            'malaria',
            with_info=True,
            as_supervised=True,
            data_dir=LOCAL_CACHE  # Download to local temp
        )
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"  Dataset size: {info.splits['train'].num_examples:,} images")
        
        # Create persistent directory on Google Drive
        DRIVE_DATASET = f"{DRIVE_DATASET_PATH}/dataset"
        os.makedirs(f"{DRIVE_DATASET}/Parasitized", exist_ok=True)
        os.makedirs(f"{DRIVE_DATASET}/Uninfected", exist_ok=True)
        
        # Function to save images to Drive
        def save_images_to_drive(dataset, prefix):
            count = 0
            for image, label in dataset:
                if label.numpy() == 1:  # Parasitized
                    save_path = f"{DRIVE_DATASET}/Parasitized/{prefix}_{count}.png"
                else:  # Uninfected
                    save_path = f"{DRIVE_DATASET}/Uninfected/{prefix}_{count}.png"
                
                pil_image = tf.keras.utils.array_to_img(image)
                pil_image.save(save_path)
                count += 1
                
                if count % 1000 == 0:
                    print(f"  Saved {count} images to Drive...")
            
            return count
        
        # Save to Google Drive
        print("\nSaving images to Google Drive (persistent storage)...")
        train_dataset = dataset['train']
        train_count = save_images_to_drive(train_dataset, 'train')
        
        # Also create a symlink from working directory to Drive for easy access
        if os.path.exists("dataset"):
            os.system("rm -rf dataset")  # Remove old symlink/dir
        os.symlink(f"{DRIVE_DATASET}", "dataset")
        print(f"✓ Created symlink: dataset -> {DRIVE_DATASET}")
        
        # Cleanup local cache
        shutil.rmtree(LOCAL_CACHE)
        
        print(f"\n✓ Dataset saved to Google Drive!")
        print(f"  Location: {DRIVE_DATASET_PATH}")
        print(f"  Total Parasitized: {len(os.listdir(f'{DRIVE_DATASET}/Parasitized'))}")
        print(f"  Total Uninfected: {len(os.listdir(f'{DRIVE_DATASET}/Uninfected'))}")
        
        return True
        
    except ImportError:
        print("TensorFlow Datasets not installed. Installing...")
        !pip install tensorflow-datasets
        return download_with_tensorflow_datasets()
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def download_with_kaggle_to_drive():
    """Download Kaggle dataset directly to Google Drive"""
    try:
        # Mount Drive
        mount_google_drive()
        
        # Set persistent path
        DRIVE_DATASET_PATH = "/content/drive/MyDrive/malarai_dataset"
        os.makedirs(DRIVE_DATASET_PATH, exist_ok=True)
        
        # Install kaggle
        !pip install kaggle -q
        
        # Configure Kaggle API
        if not os.path.exists("/root/.kaggle/kaggle.json"):
            print("\n⚠ Kaggle API not configured.")
            print("Please upload your kaggle.json file:")
            
            from google.colab import files
            uploaded = files.upload()
            
            os.makedirs("/root/.kaggle", exist_ok=True)
            for filename in uploaded.keys():
                shutil.move(filename, "/root/.kaggle/kaggle.json")
            os.chmod("/root/.kaggle/kaggle.json", 600)
        
        # Download to Drive
        print("Downloading to Google Drive...")
        os.chdir(DRIVE_DATASET_PATH)
        
        !kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile("cell-images-for-detecting-malaria.zip", 'r') as zip_ref:
            zip_ref.extractall("temp_extract")
        
        # Organize
        os.makedirs("dataset/Parasitized", exist_ok=True)
        os.makedirs("dataset/Uninfected", exist_ok=True)
        shutil.move("temp_extract/cell_images/Parasitized/*", "dataset/Parasitized/")
        shutil.move("temp_extract/cell_images/Uninfected/*", "dataset/Uninfected/")
        
        # Cleanup
        os.remove("cell-images-for-detecting-malaria.zip")
        shutil.rmtree("temp_extract")
        
        # Create symlink from working directory
        if os.path.exists("/content/malarai/dataset"):
            os.system("rm -rf /content/malarai/dataset")
        os.symlink(f"{DRIVE_DATASET_PATH}/dataset", "/content/malarai/dataset")
        
        print(f"✓ Dataset saved to: {DRIVE_DATASET_PATH}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def verify_drive_dataset():
    """Verify dataset exists on Google Drive"""
    
    DRIVE_DATASET_PATH = "/content/drive/MyDrive/malarai_dataset/dataset"
    
    if not os.path.exists(DRIVE_DATASET_PATH):
        print("✗ Dataset not found in Google Drive")
        return False
    
    parasitized_path = f"{DRIVE_DATASET_PATH}/Parasitized"
    uninfected_path = f"{DRIVE_DATASET_PATH}/Uninfected"
    
    if os.path.exists(parasitized_path) and os.path.exists(uninfected_path):
        parasitized_count = len(os.listdir(parasitized_path))
        uninfected_count = len(os.listdir(uninfected_path))
        
        print(f"\n✓ Dataset verified on Google Drive:")
        print(f"  - Parasitized: {parasitized_count:,} images")
        print(f"  - Uninfected: {uninfected_count:,} images")
        print(f"  - Total: {parasitized_count + uninfected_count:,} images")
        print(f"  - Location: {DRIVE_DATASET_PATH}")
        
        # Create symlink if needed
        if not os.path.exists("dataset"):
            os.symlink(DRIVE_DATASET_PATH, "dataset")
            print("✓ Created symlink: dataset -> Google Drive")
        
        return True
    else:
        print("✗ Dataset structure incomplete")
        return False

def check_existing_dataset():
    """Check if dataset already exists on Drive"""
    
    DRIVE_DATASET_PATH = "/content/drive/MyDrive/malarai_dataset/dataset"
    
    if os.path.exists(DRIVE_DATASET_PATH):
        print("\n" + "="*50)
        print("Found existing dataset on Google Drive!")
        print("="*50)
        
        parasitized_count = len(os.listdir(f"{DRIVE_DATASET_PATH}/Parasitized"))
        uninfected_count = len(os.listdir(f"{DRIVE_DATASET_PATH}/Uninfected"))
        
        print(f"  Parasitized: {parasitized_count:,} images")
        print(f"  Uninfected: {uninfected_count:,} images")
        
        response = input("\nUse existing dataset? (y/n): ")
        if response.lower() == 'y':
            # Create symlink
            if os.path.exists("dataset"):
                os.system("rm -rf dataset")
            os.symlink(DRIVE_DATASET_PATH, "dataset")
            print("✓ Using existing dataset from Google Drive")
            return True
    
    return False

def download_dataset(method="tensorflow"):
    """Main download function with Drive persistence"""
    
    print("="*50)
    print("Malaria Dataset Download (with Drive Persistence)")
    print("="*50)
    
    # Mount Drive first
    mount_google_drive()
    
    # Check if already downloaded
    if check_existing_dataset():
        return True
    
    # Download based on method
    if method == "tensorflow":
        success = download_with_tensorflow_datasets()
    elif method == "kaggle":
        success = download_with_kaggle_to_drive()
    else:
        print("Invalid method. Available: tensorflow, kaggle")
        return False
    
    if success:
        verify_drive_dataset()
        print("\n✓ Dataset saved to Google Drive permanently!")
        print("  You won't need to download again on next run.")
        return True
    else:
        print("\n✗ Download failed.")
        return False

if __name__ == "__main__":
    # Choose your method
    download_dataset(method="tensorflow")  # or method="kaggle"
```

## 3) Training script
comment: I dont think it follows the 80:20 method of AI training
```py
"""
MalarAI — Complete Training Script with Proper Data Split & Drive Persistence
----------------------------------------------------------------------------
This script implements:
- Proper 70% train / 15% validation / 15% test split
- No data leakage between splits
- Automatic Google Drive mounting and model persistence
- Two-phase transfer learning with MobileNetV2
- Comprehensive evaluation on held-out test set

Run:
    python train_malaria.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import shutil
import seaborn as sns

# ─────────────────────────────────────────────
# 0. MOUNT GOOGLE DRIVE (for persistence)
# ─────────────────────────────────────────────

def mount_google_drive():
    """Auto-mount Google Drive if in Colab"""
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount('/content/drive')
            print("✓ Google Drive mounted")
        return True
    except ImportError:
        print("⚠ Not in Colab, skipping Drive mount")
        return False
    except Exception as e:
        print(f"⚠ Drive mount failed: {e}")
        return False

# Mount drive at start
mount_google_drive()

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

DATASET_DIR     = "dataset"              # Original dataset (Parasitized/Uninfected)
SPLIT_DIR       = "dataset_split"        # Properly split dataset
IMG_SIZE        = (224, 224)             # MobileNetV2 expects 224x224
BATCH_SIZE      = 32                     # Reduce to 16 if OOM errors
EPOCHS_PHASE1   = 10                     # Frozen base training
EPOCHS_PHASE2   = 10                     # Fine-tuning phase
MODEL_OUTPUT    = "malaria_model.h5"
TFLITE_OUTPUT   = "malaria_model.tflite"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/malarai_models/"
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
# 2. CREATE PROPER 70-15-15 DATA SPLIT
# ─────────────────────────────────────────────

def create_proper_split():
    """
    Split dataset into:
    - 70% training
    - 15% validation
    - 15% testing

    This prevents data leakage and gives proper evaluation.
    """
    print("\n" + "="*60)
    print("STEP 1: Creating Proper 70-15-15 Data Split")
    print("="*60)

    # Check if split already exists
    if os.path.exists(SPLIT_DIR) and len(os.listdir(SPLIT_DIR)) > 0:
        response = input(f"\n{SPLIT_DIR} already exists. Re-split? (y/n): ")
        if response.lower() != 'y':
            print("✓ Using existing split directory")
            return True

    # Remove existing split directory if it exists
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['Parasitized', 'Uninfected']:
            os.makedirs(f"{SPLIT_DIR}/{split}/{class_name}", exist_ok=True)

    # Split each class
    split_counts = {'train': 0, 'val': 0, 'test': 0}

    for class_name in ['Parasitized', 'Uninfected']:
        class_path = f"{DATASET_DIR}/{class_name}"
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        print(f"\n  Splitting {class_name}: {len(images)} images")

        # First split: 70% train, 30% temp (for val+test)
        train_imgs, temp_imgs = train_test_split(
            images,
            test_size=0.30,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        # Second split: 50% of temp = 15% val, 50% = 15% test
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=0.50,  # 50% of 30% = 15%
            random_state=RANDOM_SEED,
            shuffle=True
        )

        # Copy files to respective directories
        for img in train_imgs:
            shutil.copy2(
                f"{class_path}/{img}",
                f"{SPLIT_DIR}/train/{class_name}/{img}"
            )
            split_counts['train'] += 1

        for img in val_imgs:
            shutil.copy2(
                f"{class_path}/{img}",
                f"{SPLIT_DIR}/val/{class_name}/{img}"
            )
            split_counts['val'] += 1

        for img in test_imgs:
            shutil.copy2(
                f"{class_path}/{img}",
                f"{SPLIT_DIR}/test/{class_name}/{img}"
            )
            split_counts['test'] += 1

        print(f"    Train: {len(train_imgs)} ({len(train_imgs)/len(images)*100:.1f}%)")
        print(f"    Val:   {len(val_imgs)} ({len(val_imgs)/len(images)*100:.1f}%)")
        print(f"    Test:  {len(test_imgs)} ({len(test_imgs)/len(images)*100:.1f}%)")

    print(f"\n✓ Dataset split complete!")
    print(f"  Total training images:   {split_counts['train']}")
    print(f"  Total validation images: {split_counts['val']}")
    print(f"  Total test images:       {split_counts['test']}")
    print(f"  Split ratio: 70-15-15")

    return True

# ─────────────────────────────────────────────
# 3. LOAD DATA WITH PROPER SPLITS
# ─────────────────────────────────────────────

def create_data_generators():
    """Create data generators with augmentation only for training"""

    print("\n" + "="*60)
    print("STEP 2: Creating Data Generators")
    print("="*60)

    # Training generator with augmentation (ONLY for training)
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode='nearest'
    )

    # Validation generator - NO augmentation, only rescaling
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    # Test generator - NO augmentation, only rescaling
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Load training data
    train_data = train_datagen.flow_from_directory(
        f"{SPLIT_DIR}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED
    )

    # Load validation data
    val_data = val_datagen.flow_from_directory(
        f"{SPLIT_DIR}/val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
        seed=RANDOM_SEED
    )

    # Load test data (held-out for final evaluation)
    test_data = test_datagen.flow_from_directory(
        f"{SPLIT_DIR}/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
        seed=RANDOM_SEED
    )

    print(f"\n✓ Data generators created:")
    print(f"  Training samples:   {train_data.samples}")
    print(f"  Validation samples: {val_data.samples}")
    print(f"  Test samples:       {test_data.samples}")
    print(f"  Classes: {train_data.class_indices}")

    return train_data, val_data, test_data

# ─────────────────────────────────────────────
# 4. BUILD MODEL (Transfer Learning)
# ─────────────────────────────────────────────

def build_model():
    """Build MobileNetV2-based model with two-phase training"""

    print("\n" + "="*60)
    print("STEP 3: Building Model Architecture")
    print("="*60)

    # Load pre-trained base
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Phase 1: Freeze entire base
    base_model.trainable = False

    # Build complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    print(f"\n✓ Model built with {base_model.count_params():,} base parameters (frozen)")

    return model, base_model

# ─────────────────────────────────────────────
# 5. PHASE 1: TRAIN TOP LAYERS ONLY
# ─────────────────────────────────────────────

def train_phase1(model, train_data, val_data):
    """Train only the custom classification head"""

    print("\n" + "="*60)
    print(f"STEP 4: Phase 1 Training ({EPOCHS_PHASE1} epochs, base frozen)")
    print("="*60)

    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_PHASE1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "best_phase1.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
    )

    print(f"\n✓ Phase 1 complete")
    print(f"  Best validation accuracy: {max(history1.history['val_accuracy']):.4f}")

    return history1

# ─────────────────────────────────────────────
# 6. PHASE 2: FINE-TUNE TOP LAYERS
# ─────────────────────────────────────────────

def train_phase2(model, base_model, train_data, val_data):
    """Fine-tune the top layers of the base model"""

    print("\n" + "="*60)
    print(f"STEP 5: Phase 2 Training ({EPOCHS_PHASE2} epochs, fine-tuning)")
    print("="*60)

    # Unfreeze base model
    base_model.trainable = True

    # Freeze all but last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Count trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w)
                           for w in model.trainable_weights])
    print(f"  Trainable parameters after unfreezing: {trainable_params:,}")

    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_PHASE2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "best_phase2.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
    )

    print(f"\n✓ Phase 2 complete")
    print(f"  Best validation accuracy: {max(history2.history['val_accuracy']):.4f}")

    return history2

# ─────────────────────────────────────────────
# 7. EVALUATE ON TEST SET (True Generalization)
# ─────────────────────────────────────────────

def evaluate_model(model, test_data):
    """Final evaluation on held-out test set"""

    print("\n" + "="*60)
    print("STEP 6: Final Evaluation on Test Set")
    print("="*60)

    # Get predictions
    test_data.reset()
    y_pred_probs = model.predict(test_data, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_data.classes

    # Classification report
    print("\n--- Classification Report (Test Set) ---")
    target_names = list(test_data.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate (Recall for infected)
    specificity = tn / (tn + fp)  # True Negative Rate (Recall for healthy)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\n--- Detailed Metrics ---")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Sensitivity:  {sensitivity:.4f} (Recall for infected)")
    print(f"  Specificity:  {specificity:.4f} (Recall for healthy)")
    print(f"  F1 Score:     {f1_score:.4f}")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}")

    return cm, accuracy, sensitivity, specificity

# ─────────────────────────────────────────────
# 8. PLOT TRAINING CURVES
# ─────────────────────────────────────────────

def plot_training_curves(history1, history2):
    """Plot accuracy and loss curves for both phases"""

    print("\n" + "="*60)
    print("STEP 7: Generating Training Curves")
    print("="*60)

    # Combine histories
    all_acc = history1.history["accuracy"] + history2.history["accuracy"]
    all_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    all_loss = history1.history["loss"] + history2.history["loss"]
    all_val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(all_acc, 'b-', label="Train accuracy", linewidth=2)
    ax1.plot(all_val_acc, 'r-', label="Validation accuracy", linewidth=2)
    ax1.axvline(x=len(history1.history["accuracy"]), color='gray',
                linestyle='--', linewidth=2, label="Fine-tune start")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(all_loss, 'b-', label="Train loss", linewidth=2)
    ax2.plot(all_val_loss, 'r-', label="Validation loss", linewidth=2)
    ax2.axvline(x=len(history1.history["loss"]), color='gray',
                linestyle='--', linewidth=2, label="Fine-tune start")
    ax2.set_title("Model Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("✓ Training curves saved to training_curves.png")
    plt.show()

# ─────────────────────────────────────────────
# 9. SAVE MODELS AND EXPORT
# ─────────────────────────────────────────────

def export_models(model):
    """Save Keras model and convert to TFLite"""

    print("\n" + "="*60)
    print("STEP 8: Exporting Models")
    print("="*60)

    # Save Keras model
    model.save(MODEL_OUTPUT)
    size_mb = os.path.getsize(MODEL_OUTPUT) / (1024 * 1024)
    print(f"✓ Keras model saved: {MODEL_OUTPUT} ({size_mb:.1f} MB)")

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_OUTPUT, "wb") as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(TFLITE_OUTPUT) / (1024 * 1024)
    print(f"✓ TFLite model saved: {TFLITE_OUTPUT} ({tflite_size:.1f} MB)")

def save_to_google_drive():
    """Copy all models and results to Google Drive for persistence"""

    print("\n" + "="*60)
    print("STEP 9: Saving to Google Drive")
    print("="*60)

    if not os.path.exists("/content/drive"):
        print("⚠ Google Drive not mounted, skipping Drive backup")
        return

    try:
        # Create drive directory
        os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)

        # Files to backup
        files_to_backup = [
            MODEL_OUTPUT,
            TFLITE_OUTPUT,
            "best_phase1.h5",
            "best_phase2.h5",
            "training_curves.png"
        ]

        print(f"  Backing up to: {DRIVE_MODEL_DIR}")

        for file in files_to_backup:
            if os.path.exists(file):
                dest = os.path.join(DRIVE_MODEL_DIR, file)
                shutil.copy2(file, dest)
                size_mb = os.path.getsize(file) / (1024 * 1024)
                print(f"  ✓ {file} ({size_mb:.1f} MB)")
            else:
                print(f"  ⚠ {file} not found")

        # Save metrics summary
        if os.path.exists("test_metrics.txt"):
            shutil.copy2("test_metrics.txt", os.path.join(DRIVE_MODEL_DIR, "test_metrics.txt"))

        print(f"\n✓ All models backed up to Google Drive!")
        print(f"  Location: {DRIVE_MODEL_DIR}")

    except Exception as e:
        print(f"✗ Failed to backup to Drive: {e}")

def save_metrics_summary(accuracy, sensitivity, specificity, cm):
    """Save test metrics to text file"""

    with open("test_metrics.txt", "w") as f:
        f.write("Malaria Detection Model - Test Set Results\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy:     {accuracy:.4f}\n")
        f.write(f"Sensitivity:  {sensitivity:.4f}\n")
        f.write(f"Specificity:  {specificity:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("\n✓ Test metrics saved to test_metrics.txt")
    

# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    """Execute complete training pipeline"""

    print("\n" + "="*60)
    print("MALARIA DETECTION MODEL - COMPLETE TRAINING PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Phase 1 epochs: {EPOCHS_PHASE1}")
    print(f"  Phase 2 epochs: {EPOCHS_PHASE2}")
    print(f"  Data split: 70% train / 15% val / 15% test")

    # Step 1: Create proper split
    create_proper_split()

    # Step 2: Create data generators
    train_data, val_data, test_data = create_data_generators()

    # Step 3: Build model
    model, base_model = build_model()

    # Step 4: Phase 1 training
    history1 = train_phase1(model, train_data, val_data)

    # Step 5: Phase 2 training
    history2 = train_phase2(model, base_model, train_data, val_data)

    # Step 6: Evaluate on test set
    cm, accuracy, sensitivity, specificity = evaluate_model(model, test_data)

    # Step 7: Plot curves
    plot_training_curves(history1, history2)

    # Step 8: Export models
    export_models(model)

    # Step 9: Save metrics
    save_metrics_summary(accuracy, sensitivity, specificity, cm)

    # Step 10: Backup to Google Drive
    save_to_google_drive()

    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal test accuracy: {accuracy:.4f}")
    print(f"TFLite model ready for deployment: {TFLITE_OUTPUT}")
    print(f"All files backed up to Google Drive: {DRIVE_MODEL_DIR}")
    print("\nTo use in Flutter app:")
    print(f"  1. Download {TFLITE_OUTPUT} from Google Drive")
    print(f"  2. Place in assets/ folder")
    print(f"  3. Use tflite_flutter plugin for inference")

if __name__ == "__main__":
    main()
```