from typing import Dict, Any
import optuna
import copy

class TunerAgent:
    def __init__(self):
        # Initialize a real Optuna study
        self.study = optuna.create_study(direction='maximize')
        
    def suggest_treatment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggests new hyperparameters using Bayesian Optimization (Optuna).
        """
        diagnosis = state.get('diagnosis')
        current_config = state.get('config', {}) 
        
        # Ask Optuna for the next set of parameters
        trial = self.study.ask()
        
        new_config = copy.deepcopy(current_config)
        
        # We can guide the search space based on diagnosis
        if diagnosis == "OVERFITTING":
            # Focus on regularization parameters
            new_config['dropout'] = trial.suggest_float('dropout', 0.3, 0.7)
            new_config['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
            new_config['epochs'] = trial.suggest_int('epochs', 15, 70) # Increased for better convergence w/ regs
            
        elif diagnosis == "UNDERFITTING":
            # Focus on capacity and learning rate
            new_config['lr'] = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            new_config['hidden_units'] = trial.suggest_int('hidden_units', 64, 512)
            new_config['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)
            new_config['epochs'] = trial.suggest_int('epochs', 20, 70) # Needs more time to learn
            
        else: 
            # Global optimization for other cases
            new_config['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            new_config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            new_config['epochs'] = trial.suggest_int('epochs', 10, 70)

        return {
            "action": "UPDATE_CONFIG",
            "config": new_config,
            "trial_id": trial.number
        }
        
    def report_result(self, trial_id: int, value: float):
        """
        Reports the result of a trial to Optuna to update the probabilistic model.
        """
        self.study.tell(trial_id, value)
