# ♾️ ApexLoop: The Autonomous Model Doctor

**ApexLoop** is an agentic AI system that autonomously diagnoses and optimizes machine learning models. It uses a swarm of specialized agents to analyze training logs, tune hyperparameters, and clean data iteratively until a target accuracy is reached.

---

## 🚀 How to Run

### 1. Prerequisites
*   **Python 3.10+**
*   **Ollama** (for local LLM inference)

1.  **Install & Serve Ollama**:
    *   Download from [ollama.com](https://ollama.com).
    *   Run `ollama serve` in a terminal.
    *   Run `ollama pull llama3` in another terminal.

### 2. Installation
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**Note**: `requirements.txt` already includes `langchain`, `langgraph`, `torch`, `torchvision`, `optuna`, and other dependencies. You may need to install `langchain-community` and `langchain-ollama` separately:
```bash
pip install langchain-community langchain-ollama
```

### 3. Execution
To run the full verification with a ResNet18 model:
```bash
python run_real_pipeline.py
```

Alternatively, you can also run:
```bash
python apex_loop/main.py
```

---

## ⭐ Key Features

*   **🛡️ Honest Validation**: Automatically creates a "gold standard" validation set (`temp_data/val_gold_standard`) that is never touched by data augmentation, ensuring no data leakage.
*   **💾 Fault Tolerance**: Saves state to `apex_state.json` after every iteration. Crashed? Just restart and it resumes instantly.
*   **📊 Live Dashboard**: Generates `training_report.png` updated in real-time to visualize Loss vs. Accuracy.
*   **🧠 Adaptive Cleaning**: The LLM detects high noise and can trigger "Aggressive" cleaning (1.5 sigma) vs "Standard" cleaning (3 sigma).
*   **🎨 SOTA Augmentation**: Uses `TrivialAugmentWide` for state-of-the-art automatic data augmentation.

---

## 🧠 System Architecture (Deep Dive)

The system is built on **LangGraph**, which defines a cyclic workflow between agents. All agents share a common `AgentState`.

### 1. The Core State (`state.py`)
This dictionary tracks the entire lifecycle of the optimization:
*   `model_path`: Path or identifier for the PyTorch model (e.g., "resnet18").
*   `train_path`: Path to the training dataset (augmented).
*   `val_path`: Path to the validation dataset (fixed gold standard).
*   `config_path`: Path to configuration JSON file.
*   `target_accuracy`: The target validation accuracy to achieve.
*   `current_accuracy`: Current validation accuracy.
*   `history`: A list of training metrics (`train_loss`, `val_loss`, `train_acc`, `val_acc`) from every iteration.
*   `diagnosis`: The current issue identified by the RCA Agent.
*   `cleaning_strictness`: Strictness level for data cleaning ("standard" or "aggressive").
*   `treatment_plan`: The suggested fix from tuner or data agents.
*   `iteration`: Current iteration number.
*   `is_solved`: Boolean flag indicating if target accuracy is reached.

### 2. The Orchestrator (`orchestrator.py`)
This is the "Brain" that manages the control flow. It uses a **StateGraph**:

1.  **`trainer_node`**:
    *   **Action**: Runs a real PyTorch training loop for a few batches per iteration.
    *   **Logic**: Loads data using `ImageFolder`, initializes ResNet18, trains for 5-6 batches, and updates `history` with new metrics. If `current_accuracy >= target_accuracy`, it terminates.
2.  **`rca_node`** (Diagnostician):
    *   **Action**: Analyzes the `history`.
    *   **Logic**: Uses **SmartRCAAgent** with **Llama-3** via Ollama to diagnose issues:
        *   High Variance (Train >> Val) → `OVERFITTING`
        *   High Bias (Low Train & Val) → `UNDERFITTING`
        *   Noisy/Volatile Loss → `DATA_QUALITY`
3.  **`tuner_node`** (Specialist):
    *   **Action**: Triggered if diagnosis is Over/Underfitting.
    *   **Logic**: Uses **Optuna** to suggest new hyperparameters. Returns an `UPDATE_CONFIG` action with new learning rate suggestions.
4.  **`data_ops_node`** (Surgeon):
    *   **Action**: Triggered if diagnosis is `DATA_QUALITY`.
    *   **Logic**: Applies data cleaning operations like outlier removal, augmentation, or creating a cleaned dataset. Returns a `MODIFY_DATA` action.

### 3. Agent Internals

#### 🔍️ SmartRCAAgent (`agents/smart_rca_agent.py`)
*   **Technology**: `langchain`, `langchain-ollama`.
*   **Mechanism**:
    1.  Extracts the last few epochs of training history.
    2.  Constructs a prompt: *"You are an ML Engineer. Analyze these training metrics and diagnose the issue..."*
    3.  Queries the local `llama3` model via Ollama API.
    4.  Parses the LLM response to set `state['diagnosis']`.

#### 🧪 TunerAgent (`agents/tuner_agent.py`)
*   **Technology**: `optuna`.
*   **Mechanism**:
    1.  Maintains a persistent Optuna `Study` object.
    2.  When called, it suggests new hyperparamerters (learning rate, optimizer settings).
    3.  Returns a treatment plan with `UPDATE_CONFIG` action containing the new configuration.

#### 🔧 DataOpsAgent (`agents/data_agent.py`)
*   **Technology**: `cleanlab`, custom data processing.
*   **Mechanism**:
    1.  Analyzes the dataset for quality issues.
    2.  Can apply transformations like outlier removal, data augmentation, or label noise correction.
    3.  Returns a treatment plan with `MODIFY_DATA` action and path to the cleaned dataset.

---

## 📂 Project Structure

*   `apex_loop/`: Core package.
    *   `main.py`: Alternative entry point for running the optimization loop.
    *   `orchestrator.py`: The LangGraph workflow definition.
    *   `state.py`: Type definitions for `AgentState`.
    *   `data_utils.py`: Logic for splitting data into strict train/val sets.
    *   `checkpoint_utils.py`: Logic for saving/loading state (persistence).
    *   `vis_utils.py`: Utilities for generating performance plots.
    *   `agents/`:
        *   `smart_rca_agent.py`: LLM-based diagnosis using Ollama.
        *   `rca_agent.py`: Simple heuristic-based RCA (legacy).
        *   `tuner_agent.py`: Optuna-based hyperparameter optimization.
        *   `data_agent.py`: Data cleaning and augmentation (TrivialAugmentWide).
*   `run_real_pipeline.py`: Main entry script that initializes the pipeline.
*   `run_mlcc_scenario.py`: Alternative scenario runner (if exists).
*   `generate_data.py`: Script to generate synthetic MLCC dataset.
*   `data/`: Directory containing synthetic training images.
*   `mlcc_config.json`: Configuration file for MLCC scenario.
*   `mlcc_dataset.csv`: CSV metadata for MLCC dataset.
*   `requirements.txt`: Python dependencies.

---

## 🛠️ Customization

*   **Change Model**: In `run_real_pipeline.py` or `apex_loop/main.py`, modify the `model_path` in `initial_state` (e.g., "resnet50", "efficientnet").
*   **Change LLM**: In `apex_loop/orchestrator.py`, update `SmartRCAAgent(model_name="mistral")` or any other Ollama-supported model.
*   **Change Target**: Adjust the `target_accuracy` in `run_real_pipeline.py` or `apex_loop/main.py`.
*   **Max Iterations**: Modify the iteration limit in the `router` function inside `orchestrator.py` (currently set to 10).
