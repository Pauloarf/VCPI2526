import os
import shutil
import random

def fix_and_recreate_split(origin_path, train_path, val_path, split_ratio=0.2):
    """
    Creates a sequence-aware validation split from the origin_train_images.
    Ensures that frames from the same sequence are not shared between train and val.
    """
    print(f"Recreating sequence-aware split from {origin_path}")
    
    # 1. Clear existing train and val folders
    for path in [train_path, val_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    classes = sorted([d for d in os.listdir(origin_path) if os.path.isdir(os.path.join(origin_path, d))])
    
    for cls in classes:
        cls_origin_path = os.path.join(origin_path, cls)
        cls_train_path = os.path.join(train_path, cls)
        cls_val_path = os.path.join(val_path, cls)
        
        os.makedirs(cls_train_path)
        os.makedirs(cls_val_path)
        
        files = [f for f in os.listdir(cls_origin_path) if f.endswith('.ppm')]
        if not files:
            continue

        # Sequence IDs are the first 5 digits (e.g., 00000)
        all_sequences = sorted(list(set([f.split('_')[0] for f in files])))
        
        random.shuffle(all_sequences)
        
        num_val = int(len(all_sequences) * split_ratio)
        val_sequences = set(all_sequences[:num_val])
        
        for f in files:
            seq_id = f.split('_')[0]
            if seq_id in val_sequences:
                shutil.copy(os.path.join(cls_origin_path, f), os.path.join(cls_val_path, f))
            else:
                shutil.copy(os.path.join(cls_origin_path, f), os.path.join(cls_train_path, f))
                    
    print(f"Split complete. Train and Val folders created in {os.path.dirname(train_path)}")

if __name__ == "__main__":
    ORIGIN = 'datasets/origin_train_images'
    TRAIN = 'datasets/train_images'
    VAL = 'datasets/val_images'
    fix_and_recreate_split(ORIGIN, TRAIN, VAL)
