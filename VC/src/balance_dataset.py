import os
import random
import shutil
import torch
from PIL import Image
from torchvision.transforms import v2

def balance_dataset(src_path, dest_path, target_count=2000, options=None):
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
            'perspective': True
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
        aug_list.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    if options.get('perspective'):
        aug_list.append(v2.RandomPerspective(distortion_scale=0.2, p=0.5))

    # Always resize to ensure consistency
    aug_list.append(v2.Resize((32, 32)))
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
