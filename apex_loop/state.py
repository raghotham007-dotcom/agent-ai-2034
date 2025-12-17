from typing import TypedDict, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    # Inputs
    model_path: str
    train_path: str
    val_path: str
    # dataset_path: str # Deprecated in favor of explicit split
    config_path: str
    target_accuracy: float
    
    # Dynamic State
    current_accuracy: float
    history: List[Dict[str, Any]] # Training logs
    diagnosis: Optional[str] # Output of RCA
    cleaning_strictness: Optional[str] # "standard" or "aggressive"
    treatment_plan: Optional[Dict[str, Any]] # Output of Tuner/Data Agent
    iteration: int
    
    # Granular Metrics
    per_class_metrics: Optional[Dict[str, float]]
    target_classes: Optional[List[str]]
    confusion_matrix: Optional[List[List[int]]] # Classes identified by RCA as problematic
    
    # Persisted Configurations
    current_class_weights: Optional[Dict[str, float]]
    current_config: Optional[Dict[str, Any]]
    global_best_acc: Optional[float]

    # Flags
    is_solved: bool
