"""
Lunar Crater Age classification - Data Preprocessing Pipeline

This module contains all functions needed to load, preprocess, and batch
lunar crater images for training a CNN model

Use load_data and you will get:
# 1. ✅ All images loaded
# 2. ✅ Validated to 227x227
# 3. ✅ Converted to RGB if needed
# 4. ✅ Normalized (z-score)
# 5. ✅ Batched efficiently
# 6. ✅ Class imbalance handled (if requested)
# 7. ✅ Shuffled for training
# 8. ✅ Ready for CNN input
"""
import random
import numpy as np
from typing import List, Tuple, Generator
from collections import Counter
from pathlib import Path
from PIL import Image

#-------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

IMAGE_SIZE = (227, 227)
CLASS_NAMES = ["ejecta", "oldcrater", "none"]

#Z-score normalization parameters (pre-calculated from lunar dataset)
NORM_MEAN = 0.3306  #Averge moon brightness (dark grey)
NORM_STD = 0.1618   #Standard deviation of moon brightness


#-------------------------------------------------------------------------------
# FUNCTIONS - single image processing
#-------------------------------------------------------------------------------

def load_and_validate_image(path: Path) -> np.ndarray:
    """
    Load image and ensure its 227 x 227 RGB
    Returns: Numpy array shape (227,227,3)

    Args:
        Path to the .jpg file

    Returns:
        NumPy array of shape (227, 227, 3) with dtype uint8 (0-255)
    """
    with Image.open(path) as img:
        #Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert("RGB")

        #Check size
        if img.size != IMAGE_SIZE:
            raise ValueError(f"Image {path.name} is {img.size}, expected {IMAGE_SIZE}")

        return np.array(img) #Shape (227, 227, 3), dtype: uint8



def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """
    Applies z-score normalization to lunar crater images
    Converts [0, 255] range -> normalized range ~[-2, +2]
    Centers data around 0 (mean = 0, std = 1)
    Helps model to focus on crater features, not brightness

    Formula:
        1. Scale: img_float = img_array / 255.0
        2. Normalize: img_normalized = (img_float - 0.3306) / 0.1618

    Args:
        img_array: NumPy array from load_and_validate_image()

    Returns:
        Normalized array, shape (227, 227, 3), dtype:float32
    """

    #Convert to float32 and scale to [0, 1]
    img_float = image_array.astype(np.float32) / 255.0

     #Z-score normalization using our calculated statistics
    return (img_float - NORM_MEAN) / NORM_STD

def preprocess_single_image(image_path: Path) -> np.ndarray:
    """
    Complete preprocessing for one lunar image
    Combines loading, validation and normalization in one call

    Pipeline:
        1.load_and_validate_image()-
        2.normalize_image() - applies z-score normalization

    Args:
        image_path: Path to .jpg file

    Returns:
        Preprocessed image array, ready for models
        Shape: (227, 227, 3), dtype: float32
        Values normalized
    """

    #Step 1: Load and validate
    raw_image = load_and_validate_image(image_path)

    #Step 2: Normalize
    process_image = normalize_image(raw_image)

    return process_image

#-------------------------------------------------------------------------------
# FUNCTIONS - batch processing
#-------------------------------------------------------------------------------

def preprocess_batch(
    image_paths: List[Path],
    output_dtype: type = np.float32
)->np.ndarray:
    """
    Process multiple images efectively in batch

    Args:
        image_paths:List of paths to .jpg files
        output_dtype: Output data type (default: float32)

    Returns:
        Batch array shape: (batch_size, 227, 227 3)
        batch_size = len(image_paths)
    """
    batch_size = len(image_paths)
    batch_array = np.zeros((batch_size, 227, 227, 3), dtype=output_dtype)

    for i, img_path in enumerate(image_paths):
        #Use our single_image function
        processed = preprocess_single_image(img_path)
        batch_array[i] = processed

    return batch_array

#-------------------------------------------------------------------------------
# FUNCTIONS - Dataset Management
#-------------------------------------------------------------------------------

def create_balanced_subset(
    data_dir: Path,
    samples_per_class: int =358,     #Match the smallest class(ejecta)
    seed: int = 42
)->List[Tuple[Path, int]]:
    """
    Create a balance data set by downsampling the majority class.

    Args:
        data_dir: Path to folder with class subfolders (ejecta/train/none)
        samples_per_class: Number of samples per class (default to ejecta count)
        seed: Random seed for reproductibility

    Returns:
        List of(image_path, class_index) tuples
        class_index: 0=ejecta, 1=oldcrater, 2=none
    """
    random.seed(seed)
    balance_samples = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = data_dir / class_name
        all_files = list(class_path.glob("*.jpg"))

        if not all_files:
            raise FileNotFoundError(f"No images found in {class_path}")

        if class_name == "none":
            #Downsample majority class
            selected = random.sample(all_files, samples_per_class)
        else:
            selected = all_files[:min(samples_per_class, len(all_files))]

        balance_samples.extend([(img_path, class_idx) for img_path in selected])

    print(f"✅Balanced subset: {samples_per_class} samples per class")
    print(f"Total: {len(balance_samples)} images")
    print(f"Classes: {CLASS_NAMES}")

    return balance_samples

def create_weighted_sampler(
    samples: List[Tuple[Path, int]],
    seed: int = 42
)->List[Tuple[Path, int]]:
    """
    Creates a weighted dataset to handle class imbalance.
    Oversamples minority classes by duplicating samples based on class weights.
    Returns a resampled list with balanced class distribution

    Args:
        samples: List of (image_path, class_label) tuples
        seed: Random seed for reproducibility
    """

    random.seed(seed)

    #Extract labels from samples
    labels = [label for _, label in samples]

    #Count samples per class
    class_counts = Counter(labels)


    print(f"Original class distribution:")
    for class_idx in sorted(class_counts.keys()):
        class_name = CLASS_NAMES[class_idx]
        count = class_counts[class_idx]
        print(f"{class_name}: {count} samples")

    #Calculate target number of samples per class (largest class)
    max_samples = max(class_counts.values())

    #Calculate weights: target / current count
    weights = {}
    resampled_samples = []

    for class_idx in sorted(class_counts.keys()):
        class_name = CLASS_NAMES[class_idx]
        class_samples = [s for s in samples if s[1] == class_idx]

        #weigth how many times we need to replicate
        weight = max_samples / len(class_samples)

        #determine ho many samples to add
        num_needed = max_samples - len(class_samples)

        if num_needed > 0:
            #Randomly select samples ro duplicate
            additional_samples = random.choices(class_samples, k = num_needed)

            #Combine original + additional samples
            resampled_class = class_samples + additional_samples
        else:
            #If already has max_samples, just use existing ones
            resampled_class = class_samples

        weights[class_idx] = weight
        resampled_samples.extend(resampled_class)

    #Shiffle the resampled dataset
    random.shuffle(resampled_samples)

    #Count final distribution
    final_counts = Counter([label for _, label in resampled_samples])
    print(f"After weighted resampling")
    for class_idx in sorted(final_counts.keys()):
        class_name = CLASS_NAMES[class_idx]
        count = final_counts[class_idx]
        weight = weights.get(class_idx, 1.0)
        print(f"{class_name}: {count} samples (weight: {weight:.2f})")

    return resampled_samples

#-------------------------------------------------------------------------------
# FUNCTIONS - Load Data
#-------------------------------------------------------------------------------
def create_array_dataloader(
    samples: List[Tuple[Path, int]],
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Create a DataLoader that yields batches (more memory efficient)

    Args:
        samples: From create_balanced_subset() - list of (path, label)
        batch_size: Images per batch
        shuffle: Whether to shuffle data
        seed: Random seed

    Yields:
        (batch_images, batch_labels) per iteration
        - batch_images: shape (batch_size, 227, 227, 3), float32
        - batch_labels: shape (batch_size,), int32
    """
    random.seed(seed)

    if shuffle:
        random.shuffle(samples)

    # Separate paths and labels
    image_paths = [item[0] for item in samples]
    labels = np.array([item[1] for item in samples], dtype=np.int32)

    n_samples = len(samples)

    # Yield batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)

        batch_paths = image_paths[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        # Process this batch
        batch_images = preprocess_batch(batch_paths)

        yield batch_images, batch_labels

def load_data(
    data_dir: Path,
    balanced: bool = True,
    batch_size: int = 32,
    use_weighted_sampling: bool = False,
    samples_per_class: int = 358,
    seed: int = 42
) ->Generator[Tuple[np.array, np.ndarray], None, None]:
    """
    MAIN FUNCTION- What we should use for loading the data needed for the CNN.

    Loads and batches lunar crater images for training/validation.

    Args:
        data_dir: Path to data folder
        balanced: True for balanced subset (prototype), false for all data (Final training)
        use_weighted_sampling: Apply weighted sampling for imbalance data (only works when balanced = false)
        batch_size: Number of images per batch
        samples_per_class: For balanced mode only
        seed: Random seed for reproducibility

    Returns:
        Generator yielding (images, labels) batches

    """

    if balanced:
        print(f"Creating BALANCED dataset ({samples_per_class} per class)")
        samples = create_balanced_subset(data_dir, samples_per_class, seed)
        print (f"Total images: {len(samples)}")

        #Create loader for balanced data
        loader = create_array_dataloader(samples, batch_size=batch_size, shuffle=True, seed=seed)

    else:
        print(f"Creating FULL dataset (all available data)")
        samples = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_path = data_dir / class_name
            all_files = list(class_path.glob("*.jpg"))
            samples.extend([(img_path, class_idx) for img_path in all_files])

            print(f"{class_name}: {len(all_files)} images")

        print(f"Total images {len(samples)}")

        if use_weighted_sampling:
            print(f"Applying weighted sampling strategy")
            #Resample the entire dataset with weights
            resampled_samples = create_weighted_sampler(samples, seed=seed)
            loader = create_array_dataloader(resampled_samples, batch_size=batch_size, shuffle=True, seed=seed)

        else:
            #Standard imbalanced loader
            print(f"Using imbalanced data without weighting")
            loader = create_array_dataloader(samples, batch_size=batch_size, shuffle=True, seed=seed)

    return loader
