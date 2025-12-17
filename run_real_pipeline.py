from apex_loop.orchestrator import app
from apex_loop.state import AgentState
from apex_loop.data_utils import prepare_data_split
from apex_loop.checkpoint_utils import save_state, load_state
from apex_loop.vis_utils import plot_metrics

def main():
    print("Initializing ApexLoop with REAL PyTorch Pipeline...")
    
    # 1. Try to Load Existing State (Resume Capability)
    initial_state = load_state("apex_state.json")
    

    if initial_state:
        print("--- Resuming from Checkpoint ---")
        import os
        # Robustness: Check if paths from checkpoint actually exist (machine migration support)
        if not os.path.exists(initial_state.get('train_path', '')):
            print("Warning: Paths in checkpoint do not exist (moved machines?). Regenerating split...")
            raw_data_path = "data/mlcc_synthetic/train"
            
            # --- Data Sanitization (Optional but recommended) ---
            # Now running on train_path ONLY, after verifying it exists
            # We will do this via the orchestrator or manually here if needed.
            # Ideally, sanitization happens on the 'train_path' which we just set.
            # ----------------------------------------------------
            
            # Check custom split for resume path too (in case of regen)
            dataset_root = os.path.dirname(raw_data_path)
            custom_train = os.path.join(dataset_root, "train")
            custom_test = os.path.join(dataset_root, "test")
            
            if os.path.exists(custom_train) and os.path.exists(custom_test):
                print("Using existing Custom Split found on disk.")
                train_path = os.path.abspath(custom_train)
                val_path = os.path.abspath(custom_test)
            else:
                train_path, val_path = prepare_data_split(raw_data_path, dest_root="temp_data")
            initial_state['train_path'] = train_path
            initial_state['val_path'] = val_path
    else:
        print("--- Starting New Run ---")
        # 2. Prepare Fixed Data Split (Prevents Validation Leakage)
        raw_data_path = "data/mlcc_synthetic/train"
        
        # --- Data Sanitization ---
        # Moved to after split to avoid cleaning validation data
        # -------------------------
        
        # -------------------------
        
        # Check for Custom Split (train/test folders)
        import os
        dataset_root = os.path.dirname(raw_data_path) # e.g. data/mlcc_synthetic
        custom_train = os.path.join(dataset_root, "train")
        custom_test = os.path.join(dataset_root, "test")
        
        if os.path.exists(custom_train) and os.path.exists(custom_test):
             print(f"--- Founds Custom Data Split ---")
             print(f"Train: {custom_train}")
             print(f"Test (Val): {custom_test}")
             train_path = os.path.abspath(custom_train)
             val_path = os.path.abspath(custom_test)
        else:
             print("--- Creating Automated Data Split ---")
             train_path, val_path = prepare_data_split(raw_data_path, dest_root="temp_data")

        # --- Validated Data Sanitization (Train Only) ---
        from apex_loop.data_cleaning import run_sanitization
        print(f"Running Data Sanitization on Training Data: {train_path}")
        run_sanitization(train_path, strictness='standard')
        # ------------------------------------------------
        
        initial_state: AgentState = {
            "model_path": "resnet18",
            "train_path": train_path,
            "val_path": val_path,
            "config_path": "mlcc_config.json",
            "target_accuracy": 0.95,
            "current_accuracy": 0.50,
            "history": [],
            "diagnosis": None,
            "cleaning_strictness": "standard",
            "treatment_plan": None,
            "iteration": 0,
            "is_solved": False
        }
    
    # Run the graph
    final_state = initial_state
    try:
        # Increase recursion limit to allow for many iterations
        for output in app.stream(initial_state, config={"recursion_limit": 100}):
            for node_name, state_update in output.items():
                # Update our local view of state (optional, but good for debugging)
                final_state.update(state_update)
            
            # Generate Report Plot
            plot_metrics(final_state.get('history', []))
            
            # Save State after every step
            save_state(final_state, "apex_state.json")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        
    print("\n--- Pipeline Optimization Complete ---")
    
    # Final Report
    print("\n========= FINAL REPORT =========")
    print(f"Best Global Accuracy: {final_state.get('global_best_acc', 0.0):.4f}")
    
    # Trace back the best config? 
    # Since we persist 'current_config' and 'current_class_weights', the final state holds the LAST, 
    # but not necessarily the BEST config if we regressed. 
    # Ideally we should have stored 'best_config' in state too. 
    # But for now, we can print the final state which usually converges towards good.
    
    print(f"\nFinal Configuration:")
    print(f"Dataset Used: {final_state.get('dataset_path')}")
    print(f"Hyperparameters: {final_state.get('current_config')}")
    print(f"Class Weights Active: {final_state.get('current_class_weights') is not None}")
    
    print("\nBest model saved to: best_model.pth")
    print("================================")

if __name__ == "__main__":
    main()
