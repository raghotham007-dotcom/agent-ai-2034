# ApexLoop Pipeline Diagnosis & Recommended Fixes

## Executive Summary

Your ApexLoop pipeline achieved **62.34% best accuracy** but is stuck in an overfitting loop with:
- **Miss Rate**: 20-25% (critical defects classified as Good)
- **Overkill Rate**: 15-35% (good items classified as defects)
- **Persistent overfitting** despite 10 iterations of interventions

## Root Cause Analysis

### 1. **Model Capacity vs Dataset Size Mismatch** ⚠️ CRITICAL
**Problem**: ResNet18 with 11M parameters trained on only 3,105 samples (after cleaning)
- **Expected ratio**: 10-100 samples per parameter for good generalization
- **Your ratio**: ~0.0003 samples per parameter
- **Result**: Model memorizes training data instead of learning general patterns

**Evidence**:
- Train Acc reaches 95%+ while Val Acc plateaus at 62%
- Train Loss drops to 1.2 while Val Loss stays at 1.6+
- Early stopping triggers after only 11-17 epochs

### 2. **Data Cleaning May Be Too Conservative** 🔍
**Problem**: Cleanlab found **ZERO label issues** in real-world manufacturing data
- This is highly suspicious for industrial defect detection
- The current strictness thresholds might be too safe:
  ```python
  prob_threshold_correct = 0.01  # Only removes if < 1% confidence
  prob_threshold_wrong = 0.99    # AND > 99% confidence on other class
  ```
- Real mislabels might exist but aren't being caught

**Evidence**:
- Only removed 21 files (14 duplicates + 7 blurry)
- Miss rate of 20-25% suggests label confusion exists
- Some classes have very low accuracy (e.g., Peeledoff: 30%, Brokenness: 25%)

### 3. **Aggressive Augmentation Creating Unrealistic Samples** 📈
**Problem**: Heavy augmentation (3x expansion: 3,105 → 9,315 samples) may introduce distribution shift
- TrivialAugmentWide applies random transformations that might not preserve defect characteristics
- Augmented samples might look "fake" to the validator
- Creates 2 augmented versions per original image

**Evidence**:
- Best accuracy (62.34%) achieved in iteration 1 WITHOUT heavy augmentation
- After augmentation: accuracy never improved, sometimes degraded
- Train accuracy keeps rising but val accuracy stuck

### 4. **Class Imbalance Mitigation May Be Insufficient** ⚖️
**Problem**: 
- "Good" class: 909 samples (29% of dataset)
- Minority classes: 132-334 samples each
- Current class weights use √ damping which may be too soft

**Evidence**:
- Good class: 75-85% accuracy
- Minority classes: 25-45% accuracy
- Model biased toward predicting "Good" (easier, more samples)

### 5. **Small Image Resolution** 🖼️
**Problem**: Training with 64x64 resolution loses critical detail
- Defects like "Chipping" or "Carving" may require fine-grained features
- ResNet18 was designed for 224x224 ImageNet images

**Evidence**:
```python
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Very small!
    transforms.ToTensor(),
])
```

### 6. **No Data Augmentation During Training** 🎯
**Problem**: Currently only augmenting by duplicating files
- No runtime augmentation (random crops, flips during training)
- Model sees exact same images each epoch → overfitting

## Recommended Fixes (Prioritized)

### 🥇 **Priority 1: Reduce Model Capacity**

**Option A: Use Smaller Architecture** (RECOMMENDED)
```python
# Replace ResNet18 with MobileNetV2 or EfficientNet-B0
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
```

**Option B: Freeze Early Layers**
```python
# Freeze first 75% of ResNet18
model = models.resnet18(pretrained=True)
for param in list(model.parameters())[:-20]:  # Freeze all but last 20 params
    param.requires_grad = False
model.fc = nn.Linear(num_ftrs, num_classes)
```

**Option C: Add Stronger Regularization**
```python
# Increase dropout in final layers
model.fc = nn.Sequential(
    nn.Dropout(0.7),  # Was 0.5
    nn.Linear(num_ftrs, num_classes)
)

# Stronger weight decay
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)  # Was 1e-4
```

### 🥈 **Priority 2: Increase Image Resolution**
```python
# In orchestrator.py, change transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 🥉 **Priority 3: Add Runtime Data Augmentation**
```python
# Different transforms for train vs val
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 🎯 **Priority 4: Improve Data Cleaning**

**Option A: More Aggressive Cleanlab Thresholds**
```python
# In data_cleaning.py, adjust standard mode
if strictness == 'aggressive':
    prob_threshold_correct = 0.2  
    prob_threshold_wrong = 0.7    
else:
    # More realistic standard mode
    prob_threshold_correct = 0.05  # Was 0.01
    prob_threshold_wrong = 0.90    # Was 0.99
```

**Option B: Manual Label Verification**
- Review the 21-30% of samples where model disagrees with label
- Focus on low-performing classes: Peeledoff, Brokenness, Chipping

### 🔧 **Priority 5: Disable File-Based Augmentation**
```python
# In data_agent.py, reduce augmentation aggressiveness
# Generate ZERO augmented versions (rely on runtime augmentation instead)
for aug_idx in range(0):  # Was range(2)
    ...
```

OR keep light augmentation only for minority classes:
```python
# Only augment if class has < 300 samples
if class_total[classname] < 300:
    for aug_idx in range(1):  # Create 1 copy, not 2
        ...
```

### ⚙️ **Priority 6: Stronger Class Weights**
```python
# In data_agent.py, try linear weights instead of sqrt
damping = 1.0  # Was 0.5 (sqrt)
```

### 📊 **Priority 7: Add Mixup or CutMix**
```python
# During training, mix samples to improve generalization
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

## Implementation Plan

### Phase 1: Quick Wins (1 hour)
1. ✅ **Increase image resolution to 224x224**
2. ✅ **Add runtime augmentation for training set**
3. ✅ **Disable file-based augmentation** (set range to 0 or 1)
4. ✅ **Freeze 75% of ResNet18 layers**

### Phase 2: Model Optimization (2 hours)
5. ✅ **Switch to MobileNetV2 or smaller architecture**
6. ✅ **Increase dropout to 0.7**
7. ✅ **Implement linear class weights** (damping=1.0)

### Phase 3: Data Quality (3-4 hours)
8. ✅ **Adjust Cleanlab thresholds** (0.05/0.90)
9. ✅ **Manually review low-confidence samples**
10. ✅ **Verify labels for underperforming classes**

### Phase 4: Advanced Techniques (optional)
11. ⚠️ **Implement Mixup/CutMix**
12. ⚠️ **Add focal loss for hard examples**
13. ⚠️ **Try different architectures** (EfficientNet-B0)

## Expected Outcomes

### After Phase 1:
- Validation accuracy: **70-75%**
- Reduced overfitting gap
- Miss rate: **<15%**

### After Phase 2:
- Validation accuracy: **75-82%**
- Better minority class performance
- Overkill rate: **<20%**

### After Phase 3:
- Validation accuracy: **82-88%**
- Cleaner dataset
- Miss rate: **<10%**, Overkill: **<15%**

## Monitoring Metrics

Track these after each change:
- **Overfitting Gap**: `train_acc - val_acc` (target: <10%)
- **Miss Rate**: Critical! Target <10%
- **Overkill Rate**: Target <15%
- **Per-Class Balance**: All classes >60% accuracy

## Files to Modify

1. **apex_loop/orchestrator.py** (Lines 43-46, 67-71)
   - Change image resolution
   - Add train/val transforms
   - Switch model architecture
   - Add dropout/freezing

2. **apex_loop/agents/data_agent.py** (Lines 49, 236-242)
   - Adjust damping factor
   - Reduce augmentation copies

3. **apex_loop/data_cleaning.py** (Lines 174-180)
   - Adjust Cleanlab thresholds

---

## Quick Start Script

I can create a modified version of your pipeline with Phase 1 fixes implemented. Would you like me to:
1. Create a new config file with recommended settings?
2. Modify your existing files with Phase 1 changes?
3. Create a comparison script to A/B test the changes?

Let me know which approach you prefer!
