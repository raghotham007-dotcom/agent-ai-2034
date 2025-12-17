import json
import os
from typing import Dict, Any, Optional
from .state import AgentState

def save_state(state: AgentState, filename: str = "apex_state.json") -> None:
    """
    Serializes the AgentState to a JSON file.
    """
    try:
        # Create a copy to avoid modifying the original state during serialization if we needed to filter
        state_to_save = state.copy()
        
        # Ensure directory exists if path has one
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        with open(filename, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        print(f"[Checkpoint] State saved to {filename}")
    except Exception as e:
        print(f"[Checkpoint] Error saving state: {e}")

def load_state(filename: str = "apex_state.json") -> Optional[AgentState]:
    """
    Loads AgentState from a JSON file. Returns None if file doesn't exist.
    """
    if not os.path.exists(filename):
        return None
        
    try:
        with open(filename, 'r') as f:
            state = json.load(f)
        print(f"[Checkpoint] State loaded from {filename}. Resuming iteration {state.get('iteration', 0)}.")
        return state
    except Exception as e:
        print(f"[Checkpoint] Error loading state: {e}")
        return None
