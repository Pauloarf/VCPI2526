import random
import os
import shutil
import torch

from PIL import Image
import util.vcpi_util as vcpi_util
from torchvision.transforms import v2


def show_random_samples(dataset, classes, title="Dataset Samples", num_rows=3, num_cols=5):
    """
    Visualizes a grid of random samples from a given dataset.
    What: Picks random indices and displays the images with their labels.
    Why: To visually inspect the data and ensure loading/transforms are correct.
    """
    indices = random.sample(range(len(dataset)), num_rows * num_cols)
    images = [dataset[i][0] for i in indices]
    targets = [dataset[i][1] for i in indices]
    
    print(title)
    vcpi_util.show_images(num_rows, num_cols, images, targets, classes)

def create_train_val_split(origin_path="datasets/origin_train_images", train_path="datasets/train_images", val_path="datasets/val_images", split_ratio=0.2):
    """
    Creates a sequence-aware validation split from the origin_train_images.\n
    Ensures that frames from the same sequence are not shared between train and val.\n
    Stores the Validation imagea in val_images/.\n
    Stores the Training images in train_images/.\n
    If a split already exists, removes the old one and creates a new one.
    """
    
    print(f"Creating sequence-aware split from {origin_path}")
    
    # 1. Clear existing train and val folders
    for path in [train_path, val_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # 2. Extracts the existing classes (e.g: 00000, 00001, ...)
    classes = sorted([d for d in os.listdir(origin_path) if os.path.isdir(os.path.join(origin_path, d))])
    
    for cls in classes:
        # 3. Create a folder in train and val dataset for each class
        cls_origin_path = os.path.join(origin_path, cls)
        cls_train_path = os.path.join(train_path, cls)
        cls_val_path = os.path.join(val_path, cls)
        
        os.makedirs(cls_train_path)
        os.makedirs(cls_val_path)
        
        # 4. Get all the images for a class
        files = [f for f in os.listdir(cls_origin_path) if f.endswith('.ppm')]
        if not files:
            continue

        # 5. Get the id of all sequences in the class
        all_sequences = sorted(list(set([f.split('_')[0] for f in files])))
        
        # 6. Shuffle the IDs for a random selection
        random.shuffle(all_sequences)
        
        # 7. Get the sequence IDs selected to be on the validation dataset using the given split_ratio 
        num_val = int(len(all_sequences) * split_ratio)
        val_sequences = set(all_sequences[:num_val])
        
        for f in files:
            seq_id = f.split('_')[0]
            if seq_id in val_sequences:
                shutil.copy(os.path.join(cls_origin_path, f), os.path.join(cls_val_path, f))
            else:
                shutil.copy(os.path.join(cls_origin_path, f), os.path.join(cls_train_path, f))
                    
    print(f"Split complete. Train and Val folders created in {os.path.dirname(train_path)} with a split_ration of {split_ratio}")


def balance_dataset(src_path="datasets/train_images", dest_path="datasets/train_balanced", target_count=2000, options=None):
    """
    Balances the dataset by augmenting minority classes using torchvision.transforms.v2.
    
    What: This function ensures every class has at least `target_count` images.
    Why: To prevent the model from being biased towards majority classes and improve generalization.
    
    Parameters:
    - src_path: Path to the original training images (e.g., datasets/train_images).
    - dest_path: Path where the balanced dataset will be stored (e.g., datasets/train_balanced).
    - target_count: The desired number of images per class (default: 2000).
    - options: Dictionary of augmentation flags ('rotate', 'affine', 'color', 'perspective').
    """
    if options is None:
        options = {
            'rotate': True,
            'affine': True,
            'color': True,
            'perspective': True,
            'blur': True,
            'sharpness': True,
            'grayscale': True,            

            # Perlin noise may be interesting as professor said
            # Their is no noise in the v2 though, we need to do it ourselfs
            'noise': False,
        }

    print(f"Balancing dataset from {src_path} to {dest_path}")
    print(f"Target count: {target_count} images per class")
    print(f"Augmentation options: {options}")

    if os.path.exists(dest_path):
        print(f"Destination folder {dest_path} already exists. Removing it to start fresh...")
        shutil.rmtree(dest_path)
    
    os.makedirs(dest_path)

    # Define the base transformations based on options
    aug_list = []
    if options.get('rotate'):
        aug_list.append(v2.RandomRotation(degrees=15))
    if options.get('affine'):
        aug_list.append(v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
    if options.get('color'):
        aug_list.append(v2.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2))
    if options.get('perspective'):
        aug_list.append(v2.RandomPerspective(distortion_scale=0.2, p=0.5))
    if options.get('blur'):
        aug_list.append(v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3))
    if options.get('sharpness'):
        aug_list.append(v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3))
    if options.get('grayscale'):
        aug_list.append(v2.RandomGrayscale(p=0.5))

    # Always resize to ensure consistency
    # aug_list.append(v2.Resize((32, 32)))
    # We work with PIL images here for saving back to disk
    aug_transform = v2.Compose(aug_list)

    classes = sorted([d for d in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, d))])

    for cls in classes:
        src_cls_path = os.path.join(src_path, cls)
        dest_cls_path = os.path.join(dest_path, cls)
        os.makedirs(dest_cls_path)

        files = [f for f in os.listdir(src_cls_path) if f.endswith('.ppm') or f.endswith('.png')]
        num_files = len(files)
        
        # 1. Copy original files
        for f in files:
            shutil.copy(os.path.join(src_cls_path, f), os.path.join(dest_cls_path, f))

        # 2. Augment if necessary
        if num_files < target_count:
            num_to_augment = target_count - num_files
            for i in range(num_to_augment):
                original_file = random.choice(files)
                img = Image.open(os.path.join(src_cls_path, original_file))

                # Apply the composed transformations
                new_img = aug_transform(img)

                # Save as PNG
                new_filename = f"aug_{i}_{original_file.replace('.ppm', '.png')}"
                new_img.save(os.path.join(dest_cls_path, new_filename))

    print("Balancing complete.")

if __name__ == "__main__":
    SRC = 'datasets/train_images'
    DEST = 'datasets/train_balanced'
    balance_dataset(SRC, DEST, target_count=2000)
