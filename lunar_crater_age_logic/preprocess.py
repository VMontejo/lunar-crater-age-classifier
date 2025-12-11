"""
Lunar Crater Age classification - Data Preprocessing Pipeline

This module contains all functions needed to load, preprocess, and batch
lunar crater images for training a CNN model

"""
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Callable, List


#-------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

IMAGE_SIZE = (227, 227)
CLASS_NAMES = ["ejecta", "oldcrater", "none"]

#Z-score normalization parameters (pre-calculated from lunar dataset)
NORM_MEAN = 0.3306  #Average moon brightness (dark grey)
NORM_STD = 0.1618   #Standard deviation of moon brightness

MODEL_PREPROCESS = {
    'vgg16': tf.keras.applications.vgg16.preprocess_input,
    'resnet50': tf.keras.applications.resnet50.preprocess_input,
    'custom': None  # No special preprocessing
}

SUPPORTED_MODELS = ['vgg16', 'resnet50', 'custom']

#-------------------------------------------------------------------------------
# FUNCTIONS - single image processing
#-------------------------------------------------------------------------------

def load_and_validate_image_tf(file_path: tf.Tensor) -> tf.Tensor:
    """
    Load and validate image using TensorFlow ops.

    Args:
        file_path: TensorFlow string tensor with file path

    Returns:
        Image tensor (227, 227, 3), uint8
    """
    # Load image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize to target (automatically validates size)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.uint8)

    return image


def normalize_image_tf(
    image: tf.Tensor,
    normalization: str = 'simple',
    model_type: str = 'custom'
) -> tf.Tensor:
    """
    Normalize image tensor for specific model type.

    Args:
        image: TensorFlow image tensor (227, 227, 3), uint8
        normalization: 'simple', 'zscore', or 'model'
        model_type: 'vgg16', 'resnet50', or 'custom'

    Returns:
        Normalized float32 tensor
    """
    # Convert to float32
    image = tf.cast(image, tf.float32)

    if normalization == 'simple':
        return image / 255.0

    elif normalization == 'zscore':
        image_normalized = image / 255.0
        return (image_normalized - NORM_MEAN) / NORM_STD

    elif normalization == 'model':
        if model_type in ['vgg16', 'resnet50']:
            return MODEL_PREPROCESS[model_type](image)
        else:  # custom
            return image / 255.0
    else:
        raise ValueError(f"Normalization '{normalization}' not supported")


def preprocess_single_image_tf(
    file_path: tf.Tensor,
    label: tf.Tensor,
    normalization: str = 'simple',
    model_type: str = 'custom'
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Complete preprocessing pipeline for one image.

    Args:
        file_path: Tensor with file path
        label: Integer class label
        normalization: 'simple', 'zscore', or 'model'
        model_type: 'vgg16', 'resnet50', or 'custom'

    Returns:
        (image_tensor, label_tensor)
    """
    # 1. Load image
    image = load_and_validate_image_tf(file_path)

    # 2. Normalize
    image = normalize_image_tf(image, normalization, model_type)

    return image, label

# ------------------------------------------------------------------------------
# BALANCING FUNCTIONS
# ------------------------------------------------------------------------------

def create_balanced_subset_tf(
    data_dir: Path,
    subset: str = 'train',
    samples_per_class: int = 358,
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Create balanced dataset by downsampling majority class.

    Args:
        data_dir: Base directory
        subset: 'train', 'val', or 'test'
        samples_per_class: Number of samples per class
        seed: Random seed

    Returns:
        (file_paths, labels) for balanced subset
    """
    import random
    random.seed(seed)

    file_paths = []
    labels = []

    subset_path = data_dir / subset

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = subset_path / class_name
        if class_path.exists():
            image_files = list(class_path.glob("*.jpg"))

            selected = random.sample(image_files, min(samples_per_class, len(image_files)))

            file_paths.extend([str(p) for p in selected])
            labels.extend([class_idx] * len(selected))

    print(f"Balanced {subset}: {len(file_paths)} images ({samples_per_class} per class)")
    return file_paths, labels


def create_weighted_sampler_tf(
    data_dir: Path,
    subset: str = 'train',
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Create weighted dataset by oversampling minority classes.

    Args:
        data_dir: Base directory
        subset: 'train', 'val', or 'test'
        seed: Random seed

    Returns:
        (file_paths, labels) with class balancing
    """
    import random
    random.seed(seed)

    # First collect all data
    all_file_paths = []
    all_labels = []

    subset_path = data_dir / subset

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = subset_path / class_name
        if class_path.exists():
            image_files = list(class_path.glob("*.jpg"))
            all_file_paths.extend([str(p) for p in image_files])
            all_labels.extend([class_idx] * len(image_files))

    # Count classes
    from collections import Counter
    class_counts = Counter(all_labels)
    print(f"Original {subset} distribution:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"{CLASS_NAMES[class_idx]}: {count} samples")

    # Calculate weights
    max_samples = max(class_counts.values())
    resampled_paths = []
    resampled_labels = []

    for class_idx in range(len(CLASS_NAMES)):
        # Get indices for this class
        class_indices = [i for i, label in enumerate(all_labels) if label == class_idx]
        class_samples = [(all_file_paths[i], all_labels[i]) for i in class_indices]

        # How many to add?
        num_needed = max_samples - len(class_samples)

        if num_needed > 0:
            # Oversample: duplicate some samples
            additional = random.choices(class_samples, k=num_needed)
            resampled_class = class_samples + additional
        else:
            resampled_class = class_samples

        resampled_paths.extend([p for p, _ in resampled_class])
        resampled_labels.extend([l for _, l in resampled_class])

    # Shuffle
    combined = list(zip(resampled_paths, resampled_labels))
    random.shuffle(combined)
    resampled_paths, resampled_labels = zip(*combined)

    print(f"After weighted sampling: {len(resampled_paths)} images")
    return list(resampled_paths), list(resampled_labels)

# ------------------------------------------------------------------------------
# DATASET CREATION FUNCTIONS
# ------------------------------------------------------------------------------

def create_tf_dataset(
    data_dir: Path,
    subset: str = 'train',
    model_type: str = 'custom',
    normalization: str = 'simple',
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
    balanced: bool = False,
    weighted_sampling: bool = False,
    samples_per_class: int = 358
) -> Tuple[tf.data.Dataset, int]:
    """
    Create a FAST TensorFlow dataset pipeline.

    Args:
        data_dir: Base directory containing train/val/test folders
        subset: 'train', 'val', or 'test'
        model_type: 'vgg16', 'resnet50', or 'custom'
        normalization: 'simple' (/255), 'zscore', or 'model'
        batch_size: Images per batch
        shuffle: Whether to shuffle data
        seed: Random seed

    Returns:
        (dataset, num_samples) - tf.data.Dataset ready for model.fit()
    """

    # Choose sampling strategy
    if balanced:
        print(f"Using BALANCED sampling ({samples_per_class} per class)")
        file_paths, labels = create_balanced_subset_tf(
            data_dir, subset, samples_per_class, seed
        )
    elif weighted_sampling:
        print(f"Using WEIGHTED sampling")
        file_paths, labels = create_weighted_sampler_tf(data_dir, subset, seed)
    else:
        # Original: load all data
        file_paths = []
        labels = []
        subset_path = data_dir / subset

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_path = subset_path / class_name
            if class_path.exists():
                image_files = list(class_path.glob("*.jpg"))
                file_paths.extend([str(p) for p in image_files])
                labels.extend([class_idx] * len(image_files))

        print(f"{subset}: {len(file_paths)} images for {model_type}")

    if not file_paths:
        raise FileNotFoundError(f"No images found in {data_dir / subset}")

    # Create tf.data pipeline
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle and not weighted_sampling:
        shuffle_buffer = min(len(file_paths), 2000)
        dataset = dataset.shuffle(shuffle_buffer, seed=seed)

    # Map preprocessing
    dataset = dataset.map(
        lambda fp, lbl: preprocess_single_image_tf(fp, lbl, normalization, model_type),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    #dataset = dataset.cache()

    # Batch and optimize
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(file_paths)

def load_data(
    data_dir: Path,
    model_type: str = 'custom',
    normalization: str = 'simple',
    batch_size: int = 32,
    seed: int = 42,
    train_balanced: bool = False,
    train_weighted_sampling: bool = False,
    samples_per_class: int = 358
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int, int]:
    """
    MAIN FUNCTION - Load data for VGG16, ResNet50, or Custom CNN.

    Returns datasets ready for model.fit() directly.

    Args:
        data_dir: Base directory containing train/val/test folders
        model_type: 'vgg16', 'resnet50', or 'custom'
        normalization: 'simple', 'zscore', or 'model'
        batch_size: Images per batch
        seed: Random seed
        train_balance: if True returns a balance dataset
        train_weighted_sampling: If True, balance training data by oversampling minority classes
        samples_per_class: When train_balance = True, the number of samples per class = 358


    Returns:
        (train_dataset, val_dataset, test_dataset, train_count, val_count, test_count) ->tf.data.Dataset object
        a Data set object is a pipeline of data that yields samples in batches and is optimized by training.

    """

    print(f"Loading data for {model_type.upper()}")
    print(f"Normalization: {normalization}")
    print(f"Batch size: {batch_size}")
    if train_balanced:
        print(f"Training: BALANCED ({samples_per_class} per class)")
    elif train_weighted_sampling:
        print(f"Training: WEIGHTED sampling")

    # Load train dataset (with optional balancing)
    train_dataset, train_count = create_tf_dataset(
        data_dir=data_dir,
        subset='train',
        model_type=model_type,
        normalization=normalization,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        balanced=train_balanced,
        weighted_sampling=train_weighted_sampling,
        samples_per_class=samples_per_class
    )

    # Load validation dataset (always full, no balancing)
    val_dataset, val_count = create_tf_dataset(
        data_dir=data_dir,
        subset='val',
        model_type=model_type,
        normalization=normalization,
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )

    # Load test dataset (always full, no balancing)
    test_dataset, test_count = create_tf_dataset(
        data_dir=data_dir,
        subset='test',
        model_type=model_type,
        normalization=normalization,
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )

    print(f"\nData loaded:")
    print(f"Training: {train_count} images ({train_count // batch_size} batches)")
    print(f"Validation: {val_count} images ({val_count // batch_size} batches)")
    print(f"Test: {test_count} images ({test_count // batch_size} batches)")

    return train_dataset, val_dataset, test_dataset, train_count, val_count, test_count
