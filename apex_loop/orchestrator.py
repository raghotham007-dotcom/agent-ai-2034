from langgraph.graph import StateGraph, END
from typing import Dict, Any
from .state import AgentState
from .agents.smart_rca_agent import SmartRCAAgent
from .agents.tuner_agent import TunerAgent
from .agents.data_agent import DataOpsAgent
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# Initialize Agents
rca_agent = SmartRCAAgent(model_name="llama3") # Using 'Smart' agent
tuner_agent = TunerAgent()
data_agent = DataOpsAgent()



def trainer_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs a REAL PyTorch training step on the provided dataset.
    """
    iteration = state.get('iteration', 0)
    print(f"\n--- Training Loop (Iteration {iteration}) ---")
    
    # Check if data agent provided a new dataset path (ONLY updates training data)
    treatment = state.get('treatment_plan')
    if treatment and treatment.get('new_dataset_path'):
        train_path = treatment['new_dataset_path']
        print(f"Using modified training dataset: {train_path}")
    else:
        train_path = state.get('train_path')

    val_path = state.get('val_path')
    
    # 1. Load Data (Real PyTorch DataLoader)
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Loading Training Data from: {train_path}")
        print(f"Loading Validation Data from: {val_path}")
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        val_dataset = datasets.ImageFolder(val_path, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        full_dataset = train_dataset # For class names access
        
        print(f"Dataset sizes: {train_size} Train, {val_size} Val images.")
        print(f"Classes: {full_dataset.classes}")
    else:
        print(f"Warning: Dataset paths not found. Train: {train_path}, Val: {val_path}")
        return {"current_accuracy": 0.0, "history": [], "iteration": iteration + 1}

    description = treatment.get('description', '') if treatment else ''
    print(f"Treatment: {description}")

    # 2. Initialize Model (Real ResNet18)
    print("Initializing ResNet18...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))
    
    # Check for Class Weights (Persisted or New)
    weights_dict = None
    if treatment and treatment.get('action') == 'APPLY_CLASS_WEIGHTS':
        weights_dict = treatment.get('class_weights')
    elif state.get('current_class_weights'):
         weights_dict = state.get('current_class_weights')

    weights_tensor = None
    if weights_dict:
        # Sort weights by class index
        sorted_weights = [weights_dict.get(cls, 1.0) for cls in full_dataset.classes]
        weights_tensor = torch.tensor(sorted_weights, dtype=torch.float)
        print(f"Applying Class Weights: {sorted_weights}")
    
    # Loss with Label Smoothing & Weights
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    
    # Prioritize Treatment Config > Persisted Config > Default
    active_config = state.get('current_config', {}) if state.get('current_config') else {}
    if treatment and treatment.get('config'):
        active_config.update(treatment.get('config'))
    
    # Use hyperparameters
    lr = active_config.get('lr', 0.001)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # LR Scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    # 3. Run Real Epochs
    # Use epochs from config, default to 5
    num_epochs = int(active_config.get('epochs', 5))
    print(f"Starting training step ({num_epochs} Epochs, LR: {lr})...")
    
    # Track metrics
    epoch_losses = []
    best_val_loss = float('inf')
    patience = 2
    triggers = 0
    
    model.train() # Set to train mode initially
    
    for epoch in range(num_epochs):
        model.train() # Ensure train mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # FULL BATCH LOOP (No limits)
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = running_loss / (len(train_loader) if len(train_loader) > 0 else 1)
        epoch_losses.append(epoch_loss)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Quick Validation for Early Stopping & Scheduler
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
        print(f"    Val Loss: {avg_val_loss:.4f}")
        
        # Step Scheduler
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            triggers = 0
        else:
            triggers += 1
            if triggers >= patience:
                print(f"Early stop triggered at epoch {epoch+1}")
                break

    # 4. Evaluation on Validation Set
    print("Running Validation & Confusion Matrix...")
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    class_correct = {classname: 0 for classname in full_dataset.classes}
    class_total = {classname: 0 for classname in full_dataset.classes}
    
    # Confusion Matrix: Rows=True, Cols=Pred
    num_classes = len(full_dataset.classes)
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            for label, prediction in zip(labels, predicted):
                classname = full_dataset.classes[label.item()]
                # Update Per-Class
                if label == prediction:
                    class_correct[classname] += 1
                class_total[classname] += 1
                
                # Update Confusion Matrix
                conf_matrix[label.item()][prediction.item()] += 1
                
    final_val_acc = val_correct / val_total if val_total > 0 else 0
    final_val_loss = val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
    
    per_class_acc = {}
    for classname in full_dataset.classes:
        total_c = class_total[classname]
        per_class_acc[classname] = class_correct[classname] / total_c if total_c > 0 else 0.0
        
    print(f"Validation Complete. Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.4f}")
    print(f"Per-Class Val Acc: {per_class_acc}")
    print(f"Confusion Matrix: {conf_matrix}")
    
    # --- Miss & Overkill Calculation ---
    # Attempt to identify the "Good" (Non-Defect) class
    good_class_idx = None
    common_good_names = ['good', 'pass', 'ok', 'normal', 'acceptable', 'no_defect']
    
    # Case-insensitive search
    for i, cls in enumerate(full_dataset.classes):
        if cls.lower() in common_good_names:
            good_class_idx = i
            break
            
    metrics_report = {}
    
    if good_class_idx is not None:
        good_class_name = full_dataset.classes[good_class_idx]
        print(f"Identified 'Good' class: {good_class_name} (Index {good_class_idx})")
        
        # 1. Overkill (False Positive Rate): Good samples incorrectly predicted as Defect
        # Row G: All samples that ARE Good.
        # Predicted != G: Predicted as Defect.
        total_good_samples = sum(conf_matrix[good_class_idx])
        overkill_count = total_good_samples - conf_matrix[good_class_idx][good_class_idx]
        overkill_rate = overkill_count / total_good_samples if total_good_samples > 0 else 0.0
        
        metrics_report['overkill_rate'] = overkill_rate
        print(f"Overkill Rate (Good -> Defect): {overkill_rate:.2%} ({overkill_count}/{total_good_samples})")
        
        # 2. Miss (False Negative Rate): Defect samples incorrectly predicted as Good
        # Rows != G: All samples that ARE Defects.
        # Predicted == G: Predicted as Good.
        total_defect_samples = 0
        miss_count = 0
        
        for r in range(num_classes):
            if r == good_class_idx: continue
            
            row_total = sum(conf_matrix[r])
            total_defect_samples += row_total
            
            # Count how many of this Defect class were predicted as Good (Column G)
            miss_count += conf_matrix[r][good_class_idx]
            
        miss_rate = miss_count / total_defect_samples if total_defect_samples > 0 else 0.0
        metrics_report['miss_rate'] = miss_rate
        print(f"Miss Rate (Defect -> Good): {miss_rate:.2%} ({miss_count}/{total_defect_samples})")
        
    else:
        print("Warning: Could not identify a 'Good' class for Miss/Overkill metrics. (Classes: {})".format(full_dataset.classes))
        metrics_report['miss_rate'] = None
        metrics_report['overkill_rate'] = None
    
    # 5. Report & Return
    
    # 5. Report & Return
    
    # Checkpointing
    global_best = state.get('global_best_acc', 0.0) or 0.0
    if final_val_acc > global_best:
        print(f"New Best Model! Acc: {final_val_acc:.4f} > {global_best:.4f}. Saving to 'best_model.pth'...")
        torch.save(model.state_dict(), "best_model.pth")
        global_best = final_val_acc
    
    # Report actual result to Tuner
    if treatment and treatment.get('action') == 'UPDATE_CONFIG':
        trial_id = treatment.get('trial_id')
        if trial_id is not None:
             tuner_agent.report_result(trial_id, final_val_acc)
    
    new_history = {
        "train_loss": epoch_losses, 
        "val_loss": [final_val_loss],
        "train_acc": epoch_acc, 
        "val_acc": final_val_acc 
    }
    
    history = state.get('history', [])
    history.append(new_history)
    
    return {
        "current_accuracy": final_val_acc,
        "history": history,
        "iteration": iteration + 1,
        "train_path": train_path, # Update persistent state
        "per_class_metrics": per_class_acc,
        "per_class_metrics": per_class_acc,
        "confusion_matrix": conf_matrix,
        "global_best_acc": global_best
    }

def rca_node(state: AgentState) -> Dict[str, Any]:
    print("--- RCA Analysis ---")
    result = rca_agent.analyze(state)
    diagnosis = result['diagnosis']
    target_classes = result.get('target_classes', [])
    cleaning_strictness = result.get('cleaning_strictness', 'standard')
    
    print(f"Diagnosis: {diagnosis}")
    if diagnosis == "DATA_QUALITY":
        print(f"Proposed Cleaning Strictness: {cleaning_strictness}")
    
    is_solved = state['current_accuracy'] >= state['target_accuracy']
    
    return {
        "diagnosis": diagnosis,
        "target_classes": target_classes, 
        "cleaning_strictness": cleaning_strictness,
        "is_solved": is_solved
    }

def tuner_node(state: AgentState) -> Dict[str, Any]:
    print("--- Tuner Intervention ---")
    treatment = tuner_agent.suggest_treatment(state)
    
    updates = {"treatment_plan": treatment}
    
    if treatment.get('action') == 'UPDATE_CONFIG':
        # Merge new config with existing to preserve other settings if partial updates supported
        # But here tuner returns full config usually.
        updates['current_config'] = treatment.get('config')
        
    return updates

def data_ops_node(state: AgentState) -> Dict[str, Any]:
    print("--- Data Ops Intervention ---")
    treatment = data_agent.apply_treatment(state)
    
    updates = {"treatment_plan": treatment}
    
    # PERISTENCE LOGIC
    if treatment.get('action') == 'APPLY_CLASS_WEIGHTS':
        updates['current_class_weights'] = treatment.get('class_weights')
        
    return updates

def router(state: AgentState) -> str:
    if state['is_solved']:
        return "end"
    
    if state['iteration'] > 10: # Max iterations
        print("Max iterations reached.")
        return "end"
        
    diagnosis = state['diagnosis']
    
    previous_plan = state.get('treatment_plan')
    previous_action = previous_plan.get('action', 'NONE') if previous_plan else 'NONE'
    
    # Robustness: Check for NO_OP (Agent gave up)
    if previous_action == "NO_OP":
        print(f"[Router] Agent returned NO_OP. Diagnosis is {diagnosis}. forcing switch.")
        if diagnosis == "OVERFITTING":
            return "tuner" # If DataOps gave up on overfitting, try tuning
        elif diagnosis == "DATA_QUALITY":
            return "tuner" # DataOps gave up
        else:
             # If Tuner gave up (unlikely with Optuna), or both stuck
             return "end"
    
    if diagnosis == "OVERFITTING":
        # Deterministic Logic:
        # If we just tuned, try data ops.
        # If we just fixed data, try tuning.
        if previous_action == "UPDATE_CONFIG":
            print("[Router] Config update didn't solve overfitting. Switching to DataOps.")
            return "data_ops"
        elif previous_action == "MODIFY_DATA":
            print("[Router] Data modification done. Switching to Tuner.")
            return "tuner"
        else:
             # Default first step for overfitting
            return "tuner"
            
    elif diagnosis == "UNDERFITTING":
        return "tuner"
    elif diagnosis == "DATA_QUALITY":
        return "data_ops"
    else:
        return "tuner" # Default to tuning

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("trainer", trainer_node)
workflow.add_node("rca", rca_node)
workflow.add_node("tuner", tuner_node)
workflow.add_node("data_ops", data_ops_node)

workflow.set_entry_point("trainer")

workflow.add_edge("trainer", "rca")

workflow.add_conditional_edges(
    "rca",
    router,
    {
        "tuner": "tuner",
        "data_ops": "data_ops",
        "end": END
    }
)

workflow.add_edge("tuner", "trainer")
workflow.add_edge("data_ops", "trainer")

app = workflow.compile()
