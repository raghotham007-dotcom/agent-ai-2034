from apex_loop.orchestrator import app
from apex_loop.state import AgentState
import json

def main():
    print("Initializing ApexLoop for MLCC Chip Defect Classification...")
    
    # Load mock config to ensure it exists (simulating user input)
    with open("mlcc_config.json", "r") as f:
        config = json.load(f)
        
    print(f"Loaded Config: {config['model_arch']} (Classes: {config['num_classes']})")
    
    initial_state: AgentState = {
        "model_path": "resnet18_mlcc.pt",
        "dataset_path": "mlcc_dataset.csv",
        "config_path": "mlcc_config.json",
        "target_accuracy": 0.95,
        "current_accuracy": 0.55, # Starting very low (random guess is 0.2)
        "history": [],
        "diagnosis": None,
        "treatment_plan": None,
        "iteration": 0,
        "is_solved": False
    }
    
    print(f"Target Accuracy: {initial_state['target_accuracy']}")
    print(f"Initial Accuracy: {initial_state['current_accuracy']}")
    
    # Run the graph
    try:
        for output in app.stream(initial_state):
            pass
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n--- Optimization Complete ---")

if __name__ == "__main__":
    main()
