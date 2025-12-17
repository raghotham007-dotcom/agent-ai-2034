import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any

def plot_metrics(history: List[Dict[str, Any]], filename: str = "training_report.png"):
    """
    Plots training and validation metrics from the history list.
    Saves the plot to the specified filename.
    """
    if not history:
        print("[Vis] No history to plot.")
        return

    iterations = range(1, len(history) + 1)
    
    # Extract metrics (taking the last epoch's value for each iteration for simplicity, 
    # or averaging. Here we take the reported 'val_acc' and 'train_acc' which are usually final for that step)
    train_accs = [h.get('train_acc', 0) for h in history]
    val_accs = [h.get('val_acc', 0) for h in history]
    
    # For loss, we might have a list of epoch losses. Let's take the mean or last.
    # Assuming 'train_loss' is a list of losses per epoch in that iteration.
    train_losses = []
    val_losses = []
    
    for h in history:
        t_loss = h.get('train_loss', [0])
        v_loss = h.get('val_loss', [0])
        
        # Take the average loss of the iteration
        train_losses.append(sum(t_loss) / len(t_loss) if t_loss else 0)
        val_losses.append(sum(v_loss) / len(v_loss) if v_loss else 0)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_accs, 'b-o', label='Train Acc')
    plt.plot(iterations, val_accs, 'g-o', label='Val Acc')
    plt.title('Accuracy over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(iterations, train_losses, 'b--o', label='Train Loss')
    plt.plot(iterations, val_losses, 'g--o', label='Val Loss')
    plt.title('Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    try:
        plt.savefig(filename)
        print(f"[Vis] Training report saved to {filename}")
    except Exception as e:
        print(f"[Vis] Error saving plot: {e}")
    finally:
        plt.close()
