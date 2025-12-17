
import os
import shutil
from apex_loop.data_cleaning import run_sanitization # Ensure import works

def test_metric_calculation():
    print("Testing Miss/Overkill Calculation...")
    
    # Mock Data
    classes = ["Defect_A", "Good_Part", "Defect_B"]
    # Good Index = 1
    
    # Conf Matrix (3x3)
    # Rows: Actual (Defect_A, Good, Defect_B)
    # Cols: Pred   (Defect_A, Good, Defect_B)
    
    conf_matrix = [
        [10, 2,  0], # Defect_A: 10 Correct, 2 Miss (Pred Good), 0 Error
        [ 1, 50, 4], # Good_Part: 1 Overkill (Pred A), 50 Correct, 4 Overkill (Pred B)
        [ 0,  5, 20] # Defect_B: 0 Error, 5 Miss (Pred Good), 20 Correct
    ]
    
    # Expected:
    # Good Class: "Good_Part" (Index 1)
    
    # Overkill (Good -> Defect):
    # Total Good = 1 + 50 + 4 = 55
    # Overkill Count = 1 + 4 = 5
    # Rate = 5/55 = 0.0909 (9.09%)
    
    # Miss (Defect -> Good):
    # Total Defect A = 12
    # Miss A = 2
    # Total Defect B = 25
    # Miss B = 5
    # Total Defect = 37
    # Total Miss = 7
    # Rate = 7/37 = 0.1891 (18.91%)
    
    # --- Logic from Orchestrator ---
    good_class_idx = 1
    
    total_good_samples = sum(conf_matrix[good_class_idx])
    overkill_count = total_good_samples - conf_matrix[good_class_idx][good_class_idx]
    overkill_rate = overkill_count / total_good_samples if total_good_samples > 0 else 0.0
    
    print(f"Calculated Overkill: {overkill_rate:.4f} (Expected: {5/55:.4f})")
    assert abs(overkill_rate - (5/55)) < 0.0001
    
    total_defect_samples = 0
    miss_count = 0
    num_classes = 3
    for r in range(num_classes):
        if r == good_class_idx: continue
        row_total = sum(conf_matrix[r])
        total_defect_samples += row_total
        miss_count += conf_matrix[r][good_class_idx]
        
    miss_rate = miss_count / total_defect_samples if total_defect_samples > 0 else 0.0
    
    print(f"Calculated Miss: {miss_rate:.4f} (Expected: {7/37:.4f})")
    assert abs(miss_rate - (7/37)) < 0.0001
    
    print("PASS: Metric Calculation Logic")

def test_split_logic():
    print("\nTesting Split Logic...")
    base_dir = "dummy_split_test"
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    # Case 1: Custom Split Exists
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    
    # Logic simulation
    raw_data_path = os.path.join(base_dir, "train") # Current plumbing
    dataset_root = os.path.dirname(raw_data_path)
    custom_train = os.path.join(dataset_root, "train")
    custom_test = os.path.join(dataset_root, "test")
    
    if os.path.exists(custom_train) and os.path.exists(custom_test):
        res = "CUSTOM"
    else:
        res = "AUTO"
        
    print(f"Case 1 Result: {res}")
    assert res == "CUSTOM"
    
    # Case 2: Only Train Exists
    shutil.rmtree(test_dir)
    if os.path.exists(custom_train) and os.path.exists(custom_test):
        res = "CUSTOM"
    else:
        res = "AUTO"
    
    print(f"Case 2 Result: {res}")
    assert res == "AUTO"
    
    print("PASS: Split Logic")
    
    # Cleanup
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_metric_calculation()
    test_split_logic()
