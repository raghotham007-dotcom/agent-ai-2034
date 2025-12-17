
import os
import shutil
import numpy as np
from PIL import Image
from apex_loop.data_cleaning import run_sanitization


def setup_dummy_data(root="dummy_data_clean_test"):
    if os.path.exists(root):
        shutil.rmtree(root)
    
    os.makedirs(root)
    # Need at least 2 classes and ~20 images for 5-fold CV to work reasonably well?
    # Actually Cleanlab/sklearn might complain with very few samples per class.
    # Let's try to generate enough tiny data.
    
    class_A = os.path.join(root, "circles")
    class_B = os.path.join(root, "squares")
    os.makedirs(class_A)
    os.makedirs(class_B)
    




    # Generate 50 circles (Red) with noise to prevent deduplication
    print("Generating circles...")
    for i in range(50):
        # Base red
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:] = (255, 0, 0)
        # Add random noise to make MD5 unique
        noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(arr)
        img.save(os.path.join(class_A, f"circle_{i}.jpg"))
    print("Circles generated.")

    # Generate 50 squares (Blue) with noise
    for i in range(50):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:] = (0, 0, 255)
        # Add random noise
        noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(arr)
        img.save(os.path.join(class_B, f"square_{i}.jpg"))
        

    # MISLABEL: Put a Blue Square in Circles folder (with noise to pass blur check)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:] = (0, 0, 255)
    noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img_wrong = Image.fromarray(arr)
    img_wrong.save(os.path.join(class_A, "mislabel_square.jpg"))
    
    # DUPLICATE:
    shutil.copy(os.path.join(class_A, "circle_0.jpg"), os.path.join(class_A, "duplicate_circle.jpg"))
    
    print(f"Created dummy dataset at {root}")
    return root

def verify():
    root = setup_dummy_data()
    
    print("Running Sanitization...")
    # This should print messages about finding duplicates and blur and mislabels
    run_sanitization(root)
    
    # Check results
    quarantine = os.path.join(os.path.dirname(root), "quarantine")
    
    # Check Duplicates
    dup_dir = os.path.join(quarantine, "duplicates", "circles")
    if os.path.exists(dup_dir) and len(os.listdir(dup_dir)) > 0:
        print("PASS: Duplicates found.")
    else:
        print("FAIL: Duplicates NOT found.")


    # Check Mislabels OR Outliers
    # A mislabeled image (Blue Square in Red Circles) is ALSO a semantic outlier.
    # It might be caught by either logic.
    mis_dir = os.path.join(quarantine, "mislabeled", "circles")
    out_dir = os.path.join(quarantine, "outliers", "circles")
    
    found = False
    
    if os.path.exists(mis_dir) and "mislabel_square.jpg" in os.listdir(mis_dir):
        print("PASS: Found in Mislabeled.")
        found = True
        
    if os.path.exists(out_dir) and "mislabel_square.jpg" in os.listdir(out_dir):
        print("PASS: Found in Outliers.")
        found = True
        
    if not found:
        print(f"FAIL: 'mislabel_square.jpg' not found in mislabeled OR outliers.")
        if os.path.exists(mis_dir): print(f"Mislabeled content: {os.listdir(mis_dir)}")
        if os.path.exists(out_dir): print(f"Outliers content: {os.listdir(out_dir)}")
    else:
        print("Verification SUCCESS.")

    print("Verification Complete.")

if __name__ == "__main__":
    verify()
