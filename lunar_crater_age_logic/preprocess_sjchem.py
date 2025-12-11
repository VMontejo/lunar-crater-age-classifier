"""
Lunar Crater Age classification - Data Preprocessing Pipeline

This module contains all functions needed to load, preprocess, and batch
lunar crater images for training a CNN model

Use load_data and you will get:
# 1. All images loaded
# 2. Validated to 227x227
# 3. Converted to RGB if needed
# 4. Normalized (z-score)
# 5. Batched efficiently
# 6. Class imbalance handled (if requested)
# 7. Shuffled for training
# 8. Ready for CNN input
"""
import random
import numpy as np
from typing import List, Tuple, Generator
from collections import Counter
from pathlib import Path
from PIL import Image
from PIL import ImageEnhance
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

#-------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------
IMAGE_SIZE = (227, 227)  # Target image size for CNN input
BATCH_SIZE = 32  # Default batch size
CLASS_NAMES = ["ejecta", "oldcrater", "none"]

# Z-score normalization parameters (pre-calculated from lunar dataset)
DATASET_MEAN = 85.3350  # Mean pixel value from your EDA
DATASET_STD = 40.4752   # Std deviation from your EDA

#-------------------------------------------------------------------------------
# FUNCTIONS - single image processing
#-------------------------------------------------------------------------------
def load_and_preprocess_image(image_path: str, augment: bool = False) -> np.ndarray:
    """
    Load an image from disk, resize to target size, convert to RGB if needed,
    and normalize using z-score normalization.

    Args:
        image_path (str): Path to the image file.
        augment (bool): Whether to apply data augmentation.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Data Augmentation (for training only)
    if augment:
        # Random rotation (0, 90, 180, 270 degrees - craters are rotationally invariant)
        rotation = random.choice([0, 90, 180, 270])
        img = img.rotate(rotation)

        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = Image.fromarray(np.clip(np.array(img) * factor, 0, 255).astype(np.uint8))

        #random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)


        # Random zoom (90% to 110%)
        if random.random() > 0.5:
            zoom_factor = random.uniform(0.9, 1.1)
            new_w = max(int(IMAGE_SIZE[0] * zoom_factor), IMAGE_SIZE[0])
            new_h = max(int(IMAGE_SIZE[1] * zoom_factor), IMAGE_SIZE[1])
            new_size = (new_w, new_h)

            # Resize image
            img = img.resize(new_size)

            # Center crop back to original size
            left = (img.width - IMAGE_SIZE[0]) // 2
            top = (img.height - IMAGE_SIZE[1]) // 2
            right = left + IMAGE_SIZE[0]
            bottom = top + IMAGE_SIZE[1]
            img = img.crop((left, top, right, bottom))

    ## Ensure final size is correct (only resize if needed)
    if img.size != IMAGE_SIZE:
        img = img.resize(IMAGE_SIZE)

    # Convert to numpy array (keep as 0-255 for now)
    img_array = np.array(img).astype(np.float32)

    # Z-score normalization using dataset statistics
    img_array = (img_array - DATASET_MEAN) / DATASET_STD

    return img_array

#-------------------------------------------------------------------------------
def z_score_standardization(image):
    """
    Apply z-score standardization to an image tensor.

    Args:
        image (tf.Tensor): Input image tensor.

    Returns:
        tf.Tensor: Standardized image tensor.
    """
    image = tf.cast(image, tf.float32)
    return (image - DATASET_MEAN) / DATASET_STD

#-------------------------------------------------------------------------------
# FUNCTIONS - batch processing
#-------------------------------------------------------------------------------
def preprocess_batch(
    batch_paths: List[str],
    augment: bool = False
) -> np.ndarray:
    """
    Preprocess a batch of images.

    Args:
        batch_paths (List[str]): List of image file paths.
        augment (bool): Whether to apply data augmentation.

    Returns:
        np.ndarray: Batch of preprocessed images.
    """
    images = [load_and_preprocess_image(path, augment=augment) for path in batch_paths]
    return np.array(images)

#-------------------------------------------------------------------------------
def create_array_dataloader(
    samples: List[Tuple[str, int]],
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    augment: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create a data loader generator that yields batches of images and labels.

    Args:
        samples: List of (image_path, label) tuples.
        batch_size: Size of each batch.
        shuffle: Whether to shuffle data.
        seed: Random seed for shuffling.
        augment: Whether to apply data augmentation.

    Yields:
        Tuple of (images, labels) arrays.
    """
    rng = random.Random(seed)

    # Shuffle once at start
    if shuffle:
        rng.shuffle(samples)

    # Infinite generator
    while True:
        batch_paths = []
        batch_labels = []

        for img_path, label in samples:
            batch_paths.append(img_path)
            batch_labels.append(label)

            if len(batch_paths) == batch_size:
                images = preprocess_batch(batch_paths, augment=augment)
                labels = np.array(batch_labels, dtype=np.int32)

                yield images, labels

                batch_paths = []
                batch_labels = []

        # Reshuffle for next epoch
        if shuffle:
            rng.shuffle(samples)

#-------------------------------------------------------------------------------
def load_data(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
    handle_class_imbalance: bool = True,
    soften_weights: bool = True,
    augment_train: bool = True,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], dict]:
    """
    Load and preprocess lunar crater images from a directory, returning
    train and validation datasets with optional class balancing.

    Args:
        data_dir (str): Directory containing class subdirectories.
        batch_size (int): Size of each batch.
        validation_split (float): Fraction of data to use for validation.
        handle_class_imbalance (bool): Whether to compute class weights.
        soften_weights (bool): Whether to apply square root to class weights.
        augment_train (bool): Whether to apply data augmentation to training data.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple containing:
        - train_dataset: Training dataset
        - val_dataset: Validation dataset
        - class_names: List of class names
        - class_weights_dict: Dictionary of class weights (if handle_class_imbalance=True)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Collect all image paths and labels
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(data_dir) / class_name
        for img_file in class_dir.glob('*.jpg'):  # Adjust extension if needed
            image_paths.append(str(img_file))
            labels.append(class_idx)

    # Shuffle data
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)

    # Split into train/val
    split_idx = int(len(combined) * (1 - validation_split))
    train_samples = combined[:split_idx]
    val_samples = combined[split_idx:]

    print(f"📊 Dataset Split:")
    print(f"   Total samples: {len(combined)}")
    print(f"   Training: {len(train_samples)}")
    print(f"   Validation: {len(val_samples)}")

    # Handle class imbalance
    class_weights_dict = None
    if handle_class_imbalance:
        train_labels = [label for _, label in train_samples]

        # Compute class weights
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )

        class_weights_dict = dict(enumerate(class_weights_vals))

        print("\n⚖️ Original Class Weights:")
        for class_id, weight in class_weights_dict.items():
            print(f"   {CLASS_NAMES[class_id]}: {weight:.4f}")

        # Soften the class weights (square root dampening)
        if soften_weights:
            for class_id in class_weights_dict:
                original_weight = class_weights_dict[class_id]
                softened_weight = np.sqrt(original_weight)
                class_weights_dict[class_id] = softened_weight

            print("\n⚖️ Softened Class Weights (After Square Root):")
            for class_id, weight in class_weights_dict.items():
                print(f"   {CLASS_NAMES[class_id]}: {weight:.4f}")

    # Create train dataset with augmentation
    def train_generator():
        return create_array_dataloader(
            train_samples,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            augment=augment_train
        )

    # Create validation dataset without augmentation
    def val_generator():
        return create_array_dataloader(
            val_samples,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            augment=False
        )

    # Convert to TensorFlow datasets
    output_signature = (
        tf.TensorSpec(shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=output_signature
    )

    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=output_signature
    )

    # Optimize dataset performance
    train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, CLASS_NAMES, class_weights_dict

#-------------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Test the data loader
    train_ds, val_ds, classes, weights = load_data(
        data_dir="/home/santanu/code/VMontejo/lunar-crater-age-classifier/raw_data/train",
        batch_size=32,
        validation_split=0.2,
        handle_class_imbalance=True,
        soften_weights=True,
        augment_train=True
    )

    print(f"\n✅ Data loaded successfully!")
    print(f"Classes: {classes}")
    print(f"Class weights: {weights}")

    # Test one batch
    for images, labels in train_ds.take(1):
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image value range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
