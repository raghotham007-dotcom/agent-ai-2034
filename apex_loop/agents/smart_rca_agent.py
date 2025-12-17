from typing import Dict, Any, List
import json
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class DiagnosisOutput(BaseModel):
    reasoning: str = Field(description="Detailed reasoning based on the training logs and per-class metrics")
    diagnosis: str = Field(description="One of: OVERFITTING, UNDERFITTING, DATA_QUALITY, FINE_TUNING_NEEDED")
    cleaning_strictness: str = Field(description="Suggested cleaning intensity if diagnosis is DATA_QUALITY or OVERFITTING. Options: 'standard' (default, 3 sigma), 'aggressive' (1.5 sigma, for severe noise/outliers).", default="standard")
    target_classes: List[str] = Field(description="List of specific class names that are problematic (e.g. 'scratch', 'dent'). Empty if global issue.", default_factory=list)

class SmartRCAAgent:
    def __init__(self, model_name: str = "llama3"):
        # Assumes Ollama is running locally on default port 11434
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=DiagnosisOutput)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Machine Learning Engineer specializing in Root Cause Analysis.
            Your goal is to diagnose why a model is not performing well based on its training history and per-class metrics.
            
            Analyze the loss curves and accuracy metrics carefully.
            - OVERFITTING: Train Loss << Val Loss, Train Acc >> Val Acc.
            - UNDERFITTING: Both Train and Val Loss are high/stagnant.
            - DATA_QUALITY: Loss is volatile, noisy, or spikes unexpectedly. Or specific classes have 0.0 accuracy while others are fine.
            - FINE_TUNING_NEEDED: Convergence is happening but slow; needs optimization.
            
            If the data seems extremely noisy (e.g., loss spikes, validation stuck despite training), suggest `cleaning_strictness='aggressive'`. Otherwise use 'standard'.
            If specific classes are underperforming (e.g., accuracy < 0.5 while others are > 0.8), identify them in `target_classes`.
            
            {format_instructions}
            """),
            ("user", """
            Model Configuration: {config}
            
            Previous Intervention: {previous_treatment}
            
            Training History (Last few epochs):
            {history}
            
            Per-Class Accuracy: {per_class_metrics}
            
            Confusion Matrix (Rows=True, Cols=Pred):
            {confusion_matrix}
            
            Diagnose the issue. Did the previous intervention help?
            """)
        ])

    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses LLM to analyze the training history.
        """
        history = state.get('history', [])
        if not history:
            return {"diagnosis": "UNKNOWN", "target_classes": []}
            
        config = state.get('config_path', 'unknown_config')
        per_class_metrics = state.get('per_class_metrics', {})
        previous_treatment = state.get('treatment_plan', 'None (First Iteration)')
        confusion_matrix = state.get('confusion_matrix', 'Not Available')
        
        # Summarize last 5 epochs for context
        recent_history = history[-5:]
        history_str = json.dumps(recent_history, indent=2)
        
        # Construct Chain
        chain = self.prompt_template | self.llm | self.parser
        
        try:
            print(f"\n[SmartRCA] Consulting Expert (LLM)...")
            result = chain.invoke({
                "config": config,
                "previous_treatment": str(previous_treatment),
                "history": history_str,
                "per_class_metrics": json.dumps(per_class_metrics, indent=2),
                "confusion_matrix": str(confusion_matrix),
                "format_instructions": self.parser.get_format_instructions()
            })
            
            print(f"[SmartRCA] Reasoning: {result['reasoning']}")
            print(f"[SmartRCA] Diagnosis: {result['diagnosis']}")
            if result.get('target_classes'):
                print(f"[SmartRCA] Target Classes: {result['target_classes']}")
            
            return {
                "diagnosis": result['diagnosis'],
                "cleaning_strictness": result.get('cleaning_strictness', 'standard'),
                "target_classes": result.get('target_classes', [])
            }
            
        except Exception as e:
            print(f"[SmartRCA] Error calling LLM: {e}. Fallback to heuristics.")
            return {"diagnosis": "UNKNOWN", "target_classes": []}
