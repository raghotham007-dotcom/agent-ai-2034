"""
Quick Test Script to Verify Phase 1 Changes

Run this before executing the full pipeline to ensure changes are applied correctly.
"""

import sys
import os

def check_imports():
    """Verify all required packages are available"""
    print("\n" + "="*60)
    print("1. CHECKING IMPORTS")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except:
        print("❌ PyTorch not found!")
        return False
        
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except:
        print("❌ TorchVision not found!")
        return False
        
    try:
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(pretrained=False)
        print(f"✅ MobileNetV2 available")
    except Exception as e:
        print(f"❌ MobileNetV2 error: {e}")
        return False
        
    return True

def check_config_values():
    """Check if configuration files have correct values"""
    print("\n" + "="*60)
    print("2. CHECKING CONFIGURATION VALUES")
    print("="*60)
    
    # Check orchestrator.py for resolution
    with open('apex_loop/orchestrator.py', 'r') as f:
        content = f.read()
        
        if 'Resize((224, 224))' in content:
            print("✅ Image resolution: 224x224 (CORRECT)")
        elif 'Resize((64, 64))' in content:
            print("❌ Image resolution still 64x64 (OLD - Phase 1 not applied!)")
            return False
        else:
            print("⚠️ Cannot determine image resolution")
            
        if 'mobilenet_v2' in content:
            print("✅ Model: MobileNetV2 (CORRECT)")
        elif 'resnet18' in content:
            print("❌ Model still ResNet18 (OLD - Phase 1 not applied!)")
            return False
        else:
            print("⚠️ Cannot determine model architecture")
            
        if 'RandomHorizontalFlip' in content or 'RandomVerticalFlip' in content:
            print("✅ Runtime augmentation: ENABLED (CORRECT)")
        else:
            print("⚠️ Runtime augmentation may not be enabled")
            
        if 'weight_decay' in content and '1e-3' in content:
            print("✅ Weight decay: 1e-3 (CORRECT)")
        else:
            print("⚠️ Weight decay may not be configured correctly")
    
    # Check data_agent.py for damping
    with open('apex_loop/agents/data_agent.py', 'r') as f:
        content = f.read()
        
        if 'damping = 0.8' in content:
            print("✅ Class weight damping: 0.8 (CORRECT)")
        elif 'damping = 0.5' in content:
            print("❌ Class weight damping still 0.5 (OLD - Phase 1 not applied!)")
            return False
        else:
            print("⚠️ Cannot determine damping value")
            
        if 'range(0)' in content:
            print("✅ File augmentation: DISABLED (CORRECT)")
        elif 'range(2)' in content:
            print("❌ File augmentation still enabled at range(2) (OLD)")
            return False
        else:
            print("⚠️ Cannot determine augmentation setting")
    
    # Check data_cleaning.py for cleanlab thresholds
    with open('apex_loop/data_cleaning.py', 'r') as f:
        content = f.read()
        
        if 'prob_threshold_correct = 0.05' in content:
            print("✅ Cleanlab threshold (correct): 0.05 (CORRECT)")
        elif 'prob_threshold_correct = 0.01' in content:
            print("❌ Cleanlab threshold still 0.01 (OLD - Phase 1 not applied!)")
            return False
        else:
            print("⚠️ Cannot determine cleanlab threshold")
            
        if 'prob_threshold_wrong = 0.90' in content:
            print("✅ Cleanlab threshold (wrong): 0.90 (CORRECT)")
        elif 'prob_threshold_wrong = 0.99' in content:
            print("❌ Cleanlab threshold still 0.99 (OLD)")
            return False
            
    return True

def estimate_training_time():
    """Estimate training time based on new settings"""
    print("\n" + "="*60)
    print("3. TRAINING TIME ESTIMATES")
    print("="*60)
    
    print("With Phase 1 changes:")
    print("  - Higher resolution (64→224): ~4x slower per epoch")
    print("  - MobileNet vs ResNet18: ~1.5x faster")
    print("  - Runtime augmentation: ~1.2x slower per epoch")
    print("  - Net effect: ~3.2x slower per epoch")
    print()
    print("Per iteration:")
    print("  - Old: ~2-3 minutes per iteration")
    print("  - New: ~6-10 minutes per iteration")
    print()
    print("Full pipeline (10 iterations):")
    print("  - Estimated: 60-100 minutes (~1-1.5 hours)")
    print()
    print("💡 TIP: Start with max_iterations=3 for testing")

def check_data():
    """Check if data paths exist"""
    print("\n" + "="*60)
    print("4. CHECKING DATA PATHS")
    print("="*60)
    
    train_path = "data/mlcc_synthetic/train"
    test_path = "data/mlcc_synthetic/test"
    
    if os.path.exists(train_path):
        train_count = sum([len(files) for r, d, files in os.walk(train_path)])
        print(f"✅ Training data found: {train_path} ({train_count} files)")
    else:
        print(f"❌ Training data NOT found: {train_path}")
        return False
        
    if os.path.exists(test_path):
        test_count = sum([len(files) for r, d, files in os.walk(test_path)])
        print(f"✅ Test data found: {test_path} ({test_count} files)")
    else:
        print(f"❌ Test data NOT found: {test_path}")
        return False
        
    return True

def main():
    print("\n" + "="*60)
    print("  PHASE 1 QUICK FIXES - VERIFICATION SCRIPT")
    print("="*60)
    
    all_ok = True
    
    all_ok &= check_imports()
    all_ok &= check_config_values()
    all_ok &= check_data()
    estimate_training_time()
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Ready to run pipeline!")
        print("="*60)
        print()
        print("Run the pipeline with:")
        print("  python run_real_pipeline.py")
        print()
        print("Or for quick testing (3 iterations only):")
        print("  Modify orchestrator.py line 351: if state['iteration'] > 3:")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review above")
        print("="*60)
        print()
        print("Common fixes:")
        print("  1. Ensure all files were edited correctly")
        print("  2. Check for typos in code changes")
        print("  3. Verify you're in the correct directory")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
