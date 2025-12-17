from typing import Dict, Any, List
import os
import shutil
from pathlib import Path
import torch
from torchvision import datasets, transforms
from PIL import Image
import random

class DataOpsAgent:
    def __init__(self):
        # Use State-of-the-Art automatic augmentation
        # TrivialAugmentWide is simple, parameter-free, and effective
        self.augmentation_transforms = transforms.TrivialAugmentWide()
        
    def analyze_distribution(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyzes class distribution to detect imbalance.
        Returns a suggested treatment plan (e.g. Class Weights).
        """
        print(f"[DataOps] Analyzing class distribution at {dataset_path}...")
        
        if not os.path.exists(dataset_path):
            return {"action": "NO_OP"}
            
        try:
            dataset = datasets.ImageFolder(dataset_path)
        except Exception as e:
            print(f"[DataOps] Error loading dataset: {e}")
            return {"action": "NO_OP"}
            
        # Count samples
        class_counts = {}
        for _, class_idx in dataset.samples:
            class_name = dataset.classes[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        print(f"[DataOps] Class Counts: {class_counts}")
        
        counts = list(class_counts.values())
        if not counts: return {"action": "NO_OP"}
        
        max_count = max(counts)
        min_count = min(counts)
        total_samples = sum(counts)
        
        # Calculate weights: (Total / (NumClasses * ClassCount)) ^ damping
        # Damping = 0.5 (Square Root) softens the weights to reduce Overkill
        damping = 0.5 
        num_classes = len(class_counts)
        weights = {}
        
        print(f"[DataOps] Calculating Class Weights with Damping Factor: {damping}")
        
        for cls, count in class_counts.items():
            raw_weight = total_samples / (num_classes * count)
            dampened_weight = raw_weight ** damping
            weights[cls] = dampened_weight
            
        # Optional: Normalize so mean weight is roughly 1.0 (helps with learning rate stability)
        mean_weight = sum(weights.values()) / num_classes
        for cls in weights:
            weights[cls] /= mean_weight
            
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 1.5:
            print(f"[DataOps] Imbalance detected (Ratio: {imbalance_ratio:.2f}). Suggesting Dampened Class Weights: {weights}")
            return {
                "action": "APPLY_CLASS_WEIGHTS",
                "class_weights": weights,
                "description": f"Detected imbalance (Ratio {imbalance_ratio:.2f}). Applying smoothed (sqrt) class weights."
            }
        elif total_samples < 500:
             return {
                 "action": "SUGGEST_AUGMENTATION",
                 "description": "Dataset is balanced but small. Augmentation recommended."
             }
        else:
            return {
                "action": "NO_OP", 
                "description": "Dataset distribution looks healthy."
            }

    def apply_treatment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies real data-level treatments:
        - DATA_QUALITY: Remove duplicates and statistical outliers
        - OVERFITTING: Augment dataset with transformed images
        """
        diagnosis = state.get('diagnosis')
        target_classes = state.get('target_classes', [])
        dataset_path = state.get('train_path', 'data/mlcc_synthetic/train')
        
        # 1. Check for Imbalance First (Data Analysis Approach)
        # Only suggest weights if NOT already applied
        if not state.get('current_class_weights'):
            analysis = self.analyze_distribution(dataset_path)
            if analysis['action'] == 'APPLY_CLASS_WEIGHTS':
                return analysis
        
        if diagnosis == "DATA_QUALITY":
            strictness = state.get('cleaning_strictness', 'standard')
            sigma = 1.5 if strictness == 'aggressive' else 3.0
            print(f"[DataOps] Strictness: {strictness} (Sigma: {sigma})")
            return self._clean_dataset(dataset_path, target_classes, sigma=sigma)
        elif diagnosis == "OVERFITTING":
            return self._augment_dataset(dataset_path, target_classes)
        
        return {"action": "NO_OP"}
    
    def _clean_dataset(self, dataset_path: str, target_classes: List[str] = None, sigma: float = 3.0) -> Dict[str, Any]:
        """
        Cleans the dataset by removing duplicate images and statistical outliers (e.g. blank/noisy images).
        Uses the shared centralized logic in data_cleaning.
        """
        print(f"[DataOps] Cleaning dataset at {dataset_path}. Targets: {target_classes or 'ALL'}. Sigma: {sigma}...")
        
        if not os.path.exists(dataset_path):
            return {"action": "NO_OP"}
        
        # Create cleaned directory with versioning
        base_dir = os.path.dirname(dataset_path)
        base_name = os.path.basename(dataset_path)

        # Clean naming: remove existing suffix
        if "_cleaned_v" in base_name:
             clean_base = base_name.split("_cleaned_v")[0]
        else:
             clean_base = base_name
             
        # Find next version
        existing_versions = [d for d in os.listdir(base_dir) if clean_base + "_cleaned_v" in d]
        version = len(existing_versions) + 1
        
        cleaned_path = os.path.join(base_dir, f"{clean_base}_cleaned_v{version}")
        
        if os.path.exists(cleaned_path):
            shutil.rmtree(cleaned_path)
        
        # 1. Copy original dataset to new location first
        print(f"[DataOps] Copying data to work layer: {cleaned_path}")
        shutil.copytree(dataset_path, cleaned_path)
        
        # 2. Run Sanitization on the COPY
        # strictness derived from sigma or passed explicitly if we change signature
        strictness = 'aggressive' if sigma < 2.0 else 'standard'
        
        from ..data_cleaning import run_sanitization
        run_sanitization(cleaned_path, strictness=strictness)
        
        # Count what remains
        original_count = sum([len(files) for r, d, files in os.walk(dataset_path)])
        final_count = sum([len(files) for r, d, files in os.walk(cleaned_path)])
        # Ideally exclude quarantine folder from count if it ends up inside, 
        # but quarantine is usually created at sibling level ../quarantine
        # So cleaned_path only contains valid files.
        
        samples_removed = original_count - final_count

        return {
            "action": "MODIFY_DATA",
            "new_dataset_path": cleaned_path,
            "description": f"Ran unified cleaning (Strictness: {strictness}). Removed {samples_removed} samples. {final_count} remain.",
            "samples_removed": samples_removed,
            "samples_kept": final_count
        }
    
    def _augment_dataset(self, dataset_path: str, target_classes: List[str] = None) -> Dict[str, Any]:
        """
        Augments the dataset. Respected target_classes to only augment specific classes.
        Prevents explosion by not re-augmenting images that are already augmented.
        """
        print(f"[DataOps] Augmenting dataset at {dataset_path}. Targets: {target_classes or 'ALL'}...")
        
        if not os.path.exists(dataset_path):
            print(f"[DataOps] Warning: Path {dataset_path} does not exist.")
            return {"action": "NO_OP"}
        
        # Create augmented directory
        # Create augmented directory with versioning
        base_dir = os.path.dirname(dataset_path)
        base_name = os.path.basename(dataset_path)
        
        # Clean naming
        if "_aug_v" in base_name:
             clean_base = base_name.split("_aug_v")[0]
        else:
             clean_base = base_name
             
        # Find next version
        existing_versions = [d for d in os.listdir(base_dir) if clean_base + "_aug_v" in d]
        version = len(existing_versions) + 1
        
        augmented_path = os.path.join(base_dir, f"{clean_base}_aug_v{version}")
        
        if os.path.exists(augmented_path):
            shutil.rmtree(augmented_path)
        
        # Load dataset
        try:
            dataset = datasets.ImageFolder(dataset_path)
        except Exception as e:
            print(f"[DataOps] Error loading dataset: {e}")
            return {"action": "NO_OP"}
        
        # Create directory structure
        for class_name in dataset.classes:
            os.makedirs(os.path.join(augmented_path, class_name), exist_ok=True)
        
        samples_created = 0
        samples_copied = 0
        
        for img_path, class_idx in dataset.samples:
            class_name = dataset.classes[class_idx]
            filename = os.path.basename(img_path)
            
            # COPY original
            new_path = os.path.join(augmented_path, class_name, filename)
            shutil.copy2(img_path, new_path)
            samples_copied += 1
            
            # DECIDE TO AUGMENT
            # 1. Must be in target_classes (if specified)
            if target_classes and class_name not in target_classes:
                continue
                
            # 2. Prevent infinite explosion: Don't augment what is already an augmentation
            if "_aug" in filename:
                continue
                
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Generate 2 augmented versions per image
                for aug_idx in range(2):
                    augmented_img = self.augmentation_transforms(img)
                    
                    # Save augmented image
                    base_name = os.path.splitext(filename)[0]
                    aug_filename = f"{base_name}_aug{aug_idx}.jpg"
                    aug_path = os.path.join(augmented_path, class_name, aug_filename)
                    
                    augmented_img.save(aug_path)
                    samples_created += 1
                    
            except Exception as e:
                print(f"[DataOps] Error augmenting {img_path}: {e}")
        
        total_samples = samples_copied + samples_created
        print(f"[DataOps] Augmentation complete. Created {samples_created} new samples. Total: {total_samples}")
        
        return {
            "action": "MODIFY_DATA",
            "new_dataset_path": augmented_path,
            "description": f"Created {samples_created} augmented samples for classes {target_classes or 'ALL'}.",
            "samples_created": samples_created,
            "total_samples": total_samples
        }
