import os
import shutil
import random
from typing import Tuple

def prepare_data_split(source_dataset_path: str, dest_root: str = "temp_data") -> Tuple[str, str]:
    """
    Creates a fixed 80/20 split of the source dataset into separate 'train' and 'val' directories.
    This ensures validation metric consistency across iterations even as training data is augmented.
    
    Args:
        source_dataset_path: Path to the original dataset (e.g., 'data/mlcc_synthetic/train')
        dest_root: Root directory to create 'train' and 'val' folders in.
        
    Returns:
        (train_path, val_path): Absolute paths to the new split directories.
    """
    if not os.path.exists(source_dataset_path):
        raise FileNotFoundError(f"Source dataset not found at {source_dataset_path}")
        
    # Define new paths
    train_dest = os.path.join(dest_root, "train_v0")
    val_dest = os.path.join(dest_root, "val_gold_standard")
    
    # Clean slate if exists
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    
    print(f"[DataUtils] Creating fixed Train/Val split at {dest_root}...")
    
    # Iterate through classes
    classes = [d for d in os.listdir(source_dataset_path) if os.path.isdir(os.path.join(source_dataset_path, d))]
    
    total_train = 0
    total_val = 0
    
    for cls in classes:
        src_cls_dir = os.path.join(source_dataset_path, cls)
        train_cls_dir = os.path.join(train_dest, cls)
        val_cls_dir = os.path.join(val_dest, cls)
        
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))]
        random.shuffle(images)
        
        # 80/20 Split
        split_idx = int(len(images) * 0.8)
        # Ensure at least 1 val image if possible? Or just standard split.
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        for img in train_imgs:
            shutil.copy2(os.path.join(src_cls_dir, img), os.path.join(train_cls_dir, img))
            
        for img in val_imgs:
            shutil.copy2(os.path.join(src_cls_dir, img), os.path.join(val_cls_dir, img))
            
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        
    print(f"[DataUtils] Split Complete. Train: {total_train}, Val: {total_val}")
    
    if total_train + total_val == 0:
        print("[DataUtils] ⚠️  WARNING: No images found! Check your dataset structure.")
        print(f"[DataUtils] expected: {source_dataset_path}/<class_name>/<image.jpg>")
        print("[DataUtils] Supported extensions: .jpg, .png, .bmp, .tif, .webp")
    
    print(f"[DataUtils] Validation set at {val_dest} is now LOCKED.")
    
    return os.path.abspath(train_dest), os.path.abspath(val_dest)
