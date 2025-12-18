
import os
import hashlib
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image, ImageFilter
from typing import List, Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues

# --- Configuration ---
# Lower threshold avoids filtering smooth surfaces (was 50.0). 
# Needs to be very low to only catch completely blurry messes.
BLUR_THRESHOLD = 5.0  
DUPLICATE_HASH_ALGO = 'md5'

def calculate_md5(file_path: str, block_size: int = 4096) -> str:
    """Calculates MD5 hash of a file."""
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(block_size), b''):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return ""

def is_blurry(image_path: str, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """
    Checks if an image is blurry using the variance of the Laplacian method.
    """
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
        
        # Check for very dark images (often artifacts)
        # If mean pixel value is < 5 (out of 255), it's basically black.
        stat = np.array(img)
        if np.mean(stat) < 5.0:
            return True, 0.0 # Consider black images as "bad"
            
        # Standard Laplacian Kernel
        kernel = ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0), scale=1)
        edges = img.filter(kernel)
        
        edge_data = np.array(edges, dtype=float)
        variance = np.var(edge_data)
        
        return variance < threshold, variance
    except Exception as e:
        print(f"Error checking blur for {image_path}: {e}")
        return False, 0.0

def get_embeddings(dataset_path: str) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Extracts embeddings for all images in the dataset using a pretrained ResNet.
    Returns: (embeddings, file_paths, labels)
    """
    print("Extracting embeddings for analysis (Outliers/Mislabels)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity() 
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        if len(dataset) == 0:
             return np.array([]), [], []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset for embeddings: {e}")
        return np.array([]), [], []
        
    embeddings = []
    paths = [s[0] for s in dataset.samples]
    labels = [s[1] for s in dataset.samples]
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            embeddings.append(features.cpu().numpy())
            
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
        
    return embeddings, paths, labels

def find_outliers(embeddings: np.ndarray, labels: List[int], paths: List[str], std_dev_threshold: float = 3.5) -> List[Tuple[str, float]]:
    """
    Identifies outliers based on distance from class mean.
    Increased threshold to 3.5 (was 2.5) to be less aggressive.
    """
    outliers = []
    
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        if not indices:
            continue
            
        class_embeddings = embeddings[indices]
        centroid = np.mean(class_embeddings, axis=0)
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + (std_dev_threshold * std_dist)
        
        for i, dist in enumerate(distances):
            if dist > threshold:
                original_idx = indices[i]
                outliers.append((paths[original_idx], float(dist)))
                
    return outliers

def find_label_issues_cleanlab(embeddings: np.ndarray, labels: List[int], paths: List[str], strictness: str = 'standard') -> List[str]:
    """
    Uses Cleanlab to find label issues with SAFETY CHECKS for "miss" and "overkill" cases.
    We want to KEEP hard examples.
    """
    if len(embeddings) < 20:
        print("Dataset too small for automated label quality analysis (<20 samples).")
        return []

    print("[Cleanlab] Estimating out-of-sample probabilities...")
    
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    
    # PCA to reduce dimensionality for small data logistic regression
    clf = make_pipeline(PCA(n_components=min(20, max(1, len(embeddings)//2))), LogisticRegression(max_iter=1000, n_jobs=-1, solver='lbfgs'))
    
    try:
        pred_probs = cross_val_predict(clf, embeddings, labels, cv=5, method='predict_proba')
    except Exception as e:
        print(f"[Cleanlab] CV failed: {e}")
        return []

    print("[Cleanlab] Analyzing label quality...")
    
    # 1. Get raw suggestions from Cleanlab
    issues_idx = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence',
    )
    
    if issues_idx is None or len(issues_idx) == 0:
        return []
        
    bad_files = []
    
    # 2. Filter Recommendations (Safety Check)
    # We only want to remove samples if the model is EXTREMELY confident they are wrong
    # AND the current label probability is effectively zero.
    
    # Strictness thresholds (PHASE 1 FIX: More realistic thresholds)
    if strictness == 'aggressive':
        prob_threshold_correct = 0.15  # If prob of given label is < 15%, maybe toss it (was 0.2)
        prob_threshold_wrong = 0.75    # If prob of other class is > 75%, toss it (was 0.7)
    else:
        # Standard/Safe (default): preserve hard examples but catch obvious mislabels
        prob_threshold_correct = 0.05  # Toss if model thinks there is < 5% chance it's correct (was 0.01)
        prob_threshold_wrong = 0.90    # AND model is > 90% sure it's the other thing (was 0.99)
        
    print(f"[Cleanlab] Processing {len(issues_idx)} candidates with strictness='{strictness}'...")
    print(f"           (Thresholds: given_label_prob < {prob_threshold_correct} AND suggested_label_prob > {prob_threshold_wrong})")
    
    for idx in issues_idx:
        path = paths[idx]
        given_label = labels[idx]
        
        # Probabilities for this sample
        probs = pred_probs[idx]
        prob_of_given_label = probs[given_label]
        
        # What does the model think it is?
        predicted_label = np.argmax(probs)
        prob_of_predicted = probs[predicted_label]
        
        # LOGIC:
        # If the model thinks it IS the given label (argmax == given), cleanlab flagged it merely as 'ambiguous' or low confidence.
        # We generally want to KEEP these as "hard examples".
        if predicted_label == given_label:
            # It's correctly predicted, just maybe low margin. KEEP.
            continue
            
        # If model thinks it's something else (Potential Mislabeled)
        # We only remove if it passes the safety thresholds
        if prob_of_given_label < prob_threshold_correct and prob_of_predicted > prob_threshold_wrong:
            print(f"[Cleanlab] Flagged {os.path.basename(path)}: Given Label Prob {prob_of_given_label:.4f}, Pred Label Prob {prob_of_predicted:.4f} (Conf: {prob_of_predicted:.2f})")
            bad_files.append(path)
        else:
            # It's a "Hard Example" (Miss or Overkill candidate)
            # e.g. Label=Default, Model thinks 60% Good, 40% Defect. 
            # This is a borderline case, valuable for training. KEEP IT.
            pass
            
    return bad_files

def quarantine_files(file_list: List[str], dataset_root: str, reason: str):
    """
    Moves files to a quarantine folder.
    """
    quarantine_root = os.path.join(os.path.dirname(dataset_root), "quarantine", reason)
    
    count = 0
    for file_path in file_list:
        if not os.path.exists(file_path):
            continue
            
        try:
            rel_path = os.path.relpath(file_path, dataset_root)
        except ValueError:
            rel_path = os.path.join("unknown_class", os.path.basename(file_path))
            
        dest_path = os.path.join(quarantine_root, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        shutil.move(file_path, dest_path)
        count += 1
        
    print(f"Moved {count} files to {quarantine_root}")
    return quarantine_root

def run_sanitization(dataset_path: str, strictness: str = 'standard'):
    """
    Main entry point for sanitization.
    """
    print(f"\n[DataCleaning] Starting sanitization for: {dataset_path} (Strictness: {strictness})")
    
    # 1. Duplicates
    print("[DataCleaning] Checking for duplicates...")
    hashes = {}
    duplicates = []
    
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path = os.path.join(root, f)
                h = calculate_md5(full_path)
                if h in hashes:
                    duplicates.append(full_path)
                else:
                    hashes[h] = full_path
                    
    if duplicates:
        print(f"Found {len(duplicates)} duplicates. Quarantining...")
        quarantine_files(duplicates, dataset_path, "duplicates")
    else:
        print("No duplicates found.")
        
    # 2. Blur (With reduced threshold)
    print(f"[DataCleaning] Checking for blurry images (Threadhold={BLUR_THRESHOLD})...")
    blurry_files = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path = os.path.join(root, f)
                is_blur, score = is_blurry(full_path, threshold=BLUR_THRESHOLD)
                if is_blur:
                    blurry_files.append(full_path)
    
    if blurry_files:
        print(f"Found {len(blurry_files)} blurry/black images. Quarantining...")
        quarantine_files(blurry_files, dataset_path, "blur")
    else:
        print("No blurry images found.")

    # 3. Deep Analysis (Outliers & Mislabels)
    print("\n[DataCleaning] Running Deep Analysis (Outliers & Label Quality)...")
    emb, paths, labels = get_embeddings(dataset_path)
    
    if len(emb) > 0:
        # A. Outliers (Semantic)
        outliers = find_outliers(emb, labels, paths, std_dev_threshold=3.5)
        outlier_files = [o[0] for o in outliers]
        
        if outlier_files:
             print(f"Found {len(outlier_files)} statistical outliers.")
             if strictness == 'aggressive':
                 print("Strictness is aggressive. Quarantining outliers...")
                 quarantine_files(outlier_files, dataset_path, "outliers")
             else:
                 print("Strictness is 'standard'. Keeping outliers (logging only).")

        else:
            print("No significant outliers found.")
            
        # B. Mislabels (Cleanlab with Safety Logic)
        label_issues = find_label_issues_cleanlab(emb, labels, paths, strictness=strictness)
        
        if label_issues:
             print(f"Found {len(label_issues)} confident label errors.")
             if strictness == 'aggressive':
                 print("Strictness is aggressive. Quarantining label issues...")
                 quarantine_files(label_issues, dataset_path, "mislabeled")
             else:
                 print("Strictness is 'standard'. Keeping label issues (logging only).")
        else:
            print("No confident label issues found.")
            
    else:
        print("Could not generate embeddings (dataset empty or invalid path).")
        
    print("[DataCleaning] Sanitization Complete.\n")
