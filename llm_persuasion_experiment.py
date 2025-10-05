#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Persuasion Experiment Framework

Based on README.txt goals and Some Thoughts.txt methodology:
- Test if LLMs can persuade each other
- Compare identity disclosure vs. anonymous conditions
- Measure stance changes after seeing opponent's response
- Include fake identity scenarios

Example usage:
  python llm_persuasion_experiment.py \
      --input moral_dilemmas.json \
      --models gpt-4o-mini claude-3-sonnet \
      --iterations 5 \
      --n_dilemmas 3 \
      --out_dir persuasion_results/ \
      --max_rounds 3 \
      --identity_disclosure_prob 0.5
"""

import os
import json
import uuid
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from enum import Enum

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Data structures
# ---------------------------

class IdentityCondition(Enum):
    ANONYMOUS = "anonymous"
    DISCLOSED = "disclosed"
    FAKE_DISCLOSED = "fake_disclosed"

@dataclass
class Scenario:
    situation: str
    options: List[str]
    ethical_considerations: List[str]

@dataclass
class Dilemma:
    name: str
    scenario: Scenario

@dataclass
class ModelIdentity:
    """Represents a model's identity for disclosure experiments"""
    real_name: str
    disclosed_name: str  # What we tell the other model
    is_fake: bool = False

@dataclass
class DialogueTurn:
    """Single turn in a dialogue"""
    speaker: str  # model name
    content: str
    timestamp: float
    turn_number: int

@dataclass
class PersuasionSession:
    """Complete persuasion session between two models"""
    session_id: str
    dilemma: Dilemma
    model_a: str
    model_b: str
    identity_condition: IdentityCondition
    initial_stance_a: str
    initial_stance_b: str
    final_stance_a: str
    final_stance_b: str
    dialogue: List[DialogueTurn]
    persuasion_success_a: bool  # Did A change stance?
    persuasion_success_b: bool  # Did B change stance?
    rounds_completed: int
    max_rounds: int

# ---------------------------
# Client (OpenRouter) - Enhanced for dialogue
# ---------------------------

class Client:
    """Enhanced OpenRouter client for persuasion experiments"""
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1", request_user_tag: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")
        self.base_url = base_url
        self.request_user_tag = request_user_tag

    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 800,
             top_p: float = 1.0,
             top_k: int = 0,
             min_p: float = 0.0,
             seed: Optional[int] = None,
             run_id: Optional[str] = None,
             retries: int = 2,
             timeout: int = 120) -> str:
        """Single stateless call with enhanced error handling"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "repetition_penalty": 1.0,
        }
        
        if seed is not None:
            payload["seed"] = int(seed)

        if self.request_user_tag:
            payload["user"] = str(self.request_user_tag)
        if run_id:
            payload["metadata"] = {"run_id": run_id}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://local.experiment/persuasion",
            "X-Title": f"LLM Persuasion Experiment {run_id or uuid.uuid4()}",
        }

        last_err = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < retries:
                    sleep_s = min(30, (2 ** attempt) + random.random())
                    time.sleep(sleep_s)
                else:
                    raise

        raise last_err or RuntimeError("Unknown error")

# ---------------------------
# Persuasion Experiment Framework
# ---------------------------

class PersuasionExperiment:
    """Main experiment class for LLM persuasion studies"""
    
    def __init__(self, client: Client, model_identities: Dict[str, ModelIdentity]):
        self.client = client
        self.model_identities = model_identities
        
    def create_initial_prompt(self, dilemma: Dilemma, model_name: str, is_first_speaker: bool) -> str:
        """Create initial prompt for getting model's stance"""
        prompt = f"""You are participating in a moral reasoning discussion. Please respond with a JSON object containing:

- "stance_label": EXACTLY one of these options (copy verbatim): {dilemma.scenario.options}
- "confidence": integer 0-100
- "reasoning": brief explanation of your choice

Situation: {dilemma.scenario.situation}

Return ONLY the JSON object."""
        return prompt
    
    def create_dialogue_prompt(self, 
                             dilemma: Dilemma, 
                             model_name: str, 
                             opponent_name: str,
                             identity_condition: IdentityCondition,
                             dialogue_history: List[DialogueTurn],
                             round_number: int) -> str:
        """Create prompt for dialogue turn"""
        
        # Build identity context
        identity_context = ""
        if identity_condition == IdentityCondition.DISCLOSED:
            identity_context = f"\nYou are discussing with {opponent_name}."
        elif identity_condition == IdentityCondition.FAKE_DISCLOSED:
            fake_identity = self.model_identities[opponent_name]
            identity_context = f"\nYou are discussing with {fake_identity.disclosed_name}."
        else:  # ANONYMOUS
            identity_context = "\nYou are discussing with another AI model (identity not disclosed)."
        
        # Build dialogue history
        history_text = ""
        for turn in dialogue_history:
            history_text += f"\n{turn.speaker}: {turn.content}"
        
        # Create benefits vs downsides frame (from Some Thoughts.txt)
        benefits_frame = f"""
Consider the benefits and downsides of each option:

{dilemma.scenario.options}

You have {3 - round_number} rounds left to discuss. Your goal is to:
1. Explain your reasoning clearly
2. Listen to the other's perspective  
3. See if you can find common ground or if your view changes

{identity_context}

Previous discussion:{history_text}

Please respond with a JSON object:
- "stance_label": your current stance (one of the options above)
- "confidence": integer 0-100  
- "reasoning": your response/argument
- "stance_changed": true/false (did you change from your initial position?)

Return ONLY the JSON object."""
        
        return benefits_frame
    
    def get_model_stance(self, model: str, prompt: str, run_id: str) -> Dict[str, Any]:
        """Get model's stance from prompt"""
        messages = [
            {"role": "system", "content": "You are a careful, rigorous reasoner. Only output valid JSON when asked."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=600,
                run_id=run_id
            )
            return self.parse_json_safely(response)
        except Exception as e:
            return {"error": str(e), "stance_label": "", "confidence": 0, "reasoning": ""}
    
    def parse_json_safely(self, s: str) -> Dict[str, Any]:
        """Safely parse JSON from model response"""
        s = s.strip()
        if not s.startswith("{"):
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start:end+1]
        try:
            return json.loads(s)
        except Exception:
            return {"error": f"Invalid JSON: {s[:200]}", "stance_label": "", "confidence": 0, "reasoning": ""}
    
    def run_persuasion_session(self, 
                             dilemma: Dilemma,
                             model_a: str,
                             model_b: str,
                             identity_condition: IdentityCondition,
                             max_rounds: int = 3,
                             session_id: str = None) -> PersuasionSession:
        """Run a complete persuasion session between two models"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        dialogue = []
        
        # Get initial stances
        initial_prompt_a = self.create_initial_prompt(dilemma, model_a, True)
        initial_prompt_b = self.create_initial_prompt(dilemma, model_b, False)
        
        initial_response_a = self.get_model_stance(model_a, initial_prompt_a, f"{session_id}_init_a")
        initial_response_b = self.get_model_stance(model_b, initial_prompt_b, f"{session_id}_init_b")
        
        initial_stance_a = initial_response_a.get("stance_label", "")
        initial_stance_b = initial_response_b.get("stance_label", "")
        
        # Record initial responses as dialogue turns
        dialogue.append(DialogueTurn(
            speaker=model_a,
            content=f"Initial stance: {initial_stance_a}. Reasoning: {initial_response_a.get('reasoning', '')}",
            timestamp=time.time(),
            turn_number=0
        ))
        
        dialogue.append(DialogueTurn(
            speaker=model_b,
            content=f"Initial stance: {initial_stance_b}. Reasoning: {initial_response_b.get('reasoning', '')}",
            timestamp=time.time(),
            turn_number=0
        ))
        
        # Run dialogue rounds
        current_stance_a = initial_stance_a
        current_stance_b = initial_stance_b
        
        for round_num in range(1, max_rounds + 1):
            # Model A responds
            dialogue_prompt_a = self.create_dialogue_prompt(
                dilemma, model_a, model_b, identity_condition, dialogue, round_num
            )
            response_a = self.get_model_stance(model_a, dialogue_prompt_a, f"{session_id}_r{round_num}_a")
            
            new_stance_a = response_a.get("stance_label", current_stance_a)
            dialogue.append(DialogueTurn(
                speaker=model_a,
                content=response_a.get("reasoning", ""),
                timestamp=time.time(),
                turn_number=round_num
            ))
            
            # Model B responds
            dialogue_prompt_b = self.create_dialogue_prompt(
                dilemma, model_b, model_a, identity_condition, dialogue, round_num
            )
            response_b = self.get_model_stance(model_b, dialogue_prompt_b, f"{session_id}_r{round_num}_b")
            
            new_stance_b = response_b.get("stance_label", current_stance_b)
            dialogue.append(DialogueTurn(
                speaker=model_b,
                content=response_b.get("reasoning", ""),
                timestamp=time.time(),
                turn_number=round_num
            ))
            
            # Update current stances
            current_stance_a = new_stance_a
            current_stance_b = new_stance_b
            
            # Check if either model changed stance
            if (current_stance_a != initial_stance_a or current_stance_b != initial_stance_b):
                # Could add early termination logic here
                pass
        
        # Determine persuasion success
        persuasion_success_a = (current_stance_a != initial_stance_a)
        persuasion_success_b = (current_stance_b != initial_stance_b)
        
        return PersuasionSession(
            session_id=session_id,
            dilemma=dilemma,
            model_a=model_a,
            model_b=model_b,
            identity_condition=identity_condition,
            initial_stance_a=initial_stance_a,
            initial_stance_b=initial_stance_b,
            final_stance_a=current_stance_a,
            final_stance_b=current_stance_b,
            dialogue=dialogue,
            persuasion_success_a=persuasion_success_a,
            persuasion_success_b=persuasion_success_b,
            rounds_completed=max_rounds,
            max_rounds=max_rounds
        )

# ---------------------------
# Helpers
# ---------------------------

def load_dilemmas(path: str) -> List[Dilemma]:
    """Load dilemmas from JSON file"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[Dilemma] = []
    for d in raw["moral_dilemmas"]:
        out.append(Dilemma(
            name=d["name"],
            scenario=Scenario(
                situation=d["scenario"]["situation"],
                options=list(d["scenario"]["options"]),
                ethical_considerations=list(d["scenario"]["ethical_considerations"]),
            )
        ))
    return out

def create_model_identities(models: List[str]) -> Dict[str, ModelIdentity]:
    """Create model identities for disclosure experiments"""
    identities = {}
    
    # Create fake identities for some models
    fake_identities = {
        "gpt-4o-mini": "GPT-5",
        "claude-3-sonnet": "Claude-4",
        "llama-3.1-8b-instruct": "GPT-4",
    }
    
    for model in models:
        identities[model] = ModelIdentity(
            real_name=model,
            disclosed_name=fake_identities.get(model, model),
            is_fake=fake_identities.get(model, "") != model
        )
    
    return identities

def main():
    ap = argparse.ArgumentParser(description="LLM Persuasion Experiment")
    ap.add_argument("--input", default="moral_dilemmas.json", help="Path to moral_dilemmas.json")
    ap.add_argument("--models", nargs="+", required=True, help="Two or more OpenRouter model IDs")
    ap.add_argument("--iterations", type=int, default=5, help="Number of sessions per condition")
    ap.add_argument("--n_dilemmas", type=int, default=3, help="Number of dilemmas to test")
    ap.add_argument("--out_dir", default="persuasion_results", help="Output directory")
    ap.add_argument("--max_rounds", type=int, default=3, help="Maximum dialogue rounds")
    ap.add_argument("--identity_disclosure_prob", type=float, default=0.5, help="Probability of identity disclosure")
    ap.add_argument("--fake_identity_prob", type=float, default=0.3, help="Probability of fake identity")
    ap.add_argument("--request_user_tag", type=str, default=None, help="User tag for analytics")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = ap.parse_args()
    
    # Setup
    dilemmas = load_dilemmas(args.input)
    selected = dilemmas[:args.n_dilemmas]
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    client = Client(request_user_tag=args.request_user_tag)
    model_identities = create_model_identities(args.models)
    experiment = PersuasionExperiment(client, model_identities)
    
    if args.seed is not None:
        random.seed(args.seed)
    
    # Run experiments
    sessions = []
    results_jsonl = out_dir / "persuasion_sessions.jsonl"
    
    with results_jsonl.open("w", encoding="utf-8") as f:
        for i in range(args.iterations):
            for dilemma in selected:
                for model_a in args.models:
                    for model_b in args.models:
                        if model_a == model_b:
                            continue
                        
                        # Randomly assign identity condition
                        rand = random.random()
                        if rand < args.fake_identity_prob:
                            identity_condition = IdentityCondition.FAKE_DISCLOSED
                        elif rand < args.identity_disclosure_prob + args.fake_identity_prob:
                            identity_condition = IdentityCondition.DISCLOSED
                        else:
                            identity_condition = IdentityCondition.ANONYMOUS
                        
                        session = experiment.run_persuasion_session(
                            dilemma=dilemma,
                            model_a=model_a,
                            model_b=model_b,
                            identity_condition=identity_condition,
                            max_rounds=args.max_rounds
                        )
                        
                        sessions.append(session)
                        
                        # Write to JSONL
                        session_dict = asdict(session)
                        # Convert dialogue turns to dicts
                        session_dict["dialogue"] = [asdict(turn) for turn in session.dialogue]
                        session_dict["identity_condition"] = session.identity_condition.value
                        
                        f.write(json.dumps(session_dict, ensure_ascii=False) + "\n")
                        
                        print(f"Session {len(sessions)}: {model_a} vs {model_b} on {dilemma.name} "
                              f"({identity_condition.value}) - "
                              f"A: {session.persuasion_success_a}, B: {session.persuasion_success_b}")
    
    # Analyze results
    analyze_results(sessions, out_dir)

def analyze_results(sessions: List[PersuasionSession], out_dir: Path):
    """Analyze and visualize persuasion results"""
    
    # Convert to DataFrame
    data = []
    for session in sessions:
        data.append({
            "session_id": session.session_id,
            "dilemma": session.dilemma.name,
            "model_a": session.model_a,
            "model_b": session.model_b,
            "identity_condition": session.identity_condition.value,
            "initial_stance_a": session.initial_stance_a,
            "initial_stance_b": session.initial_stance_b,
            "final_stance_a": session.final_stance_a,
            "final_stance_b": session.final_stance_b,
            "persuasion_success_a": session.persuasion_success_a,
            "persuasion_success_b": session.persuasion_success_b,
            "any_persuasion": session.persuasion_success_a or session.persuasion_success_b,
            "rounds_completed": session.rounds_completed,
        })
    
    df = pd.DataFrame(data)
    df.to_csv(out_dir / "persuasion_results.csv", index=False)
    
    # Summary statistics
    summary = df.groupby(["identity_condition", "model_a", "model_b"]).agg({
        "persuasion_success_a": "mean",
        "persuasion_success_b": "mean", 
        "any_persuasion": "mean",
        "session_id": "count"
    }).round(3)
    
    summary.to_csv(out_dir / "persuasion_summary.csv")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Persuasion rate by identity condition
    persuasion_by_condition = df.groupby("identity_condition")["any_persuasion"].mean()
    
    plt.subplot(2, 2, 1)
    persuasion_by_condition.plot(kind="bar")
    plt.title("Persuasion Rate by Identity Condition")
    plt.ylabel("Persuasion Rate")
    plt.xticks(rotation=45)
    
    # Persuasion rate by model pair
    plt.subplot(2, 2, 2)
    model_pairs = df.groupby(["model_a", "model_b"])["any_persuasion"].mean().sort_values(ascending=False)
    model_pairs.head(10).plot(kind="bar")
    plt.title("Top 10 Model Pairs by Persuasion Rate")
    plt.ylabel("Persuasion Rate")
    plt.xticks(rotation=45)
    
    # Persuasion rate by dilemma
    plt.subplot(2, 2, 3)
    dilemma_persuasion = df.groupby("dilemma")["any_persuasion"].mean().sort_values(ascending=False)
    dilemma_persuasion.plot(kind="bar")
    plt.title("Persuasion Rate by Dilemma")
    plt.ylabel("Persuasion Rate")
    plt.xticks(rotation=45)
    
    # Identity condition comparison
    plt.subplot(2, 2, 4)
    condition_comparison = df.groupby("identity_condition").agg({
        "persuasion_success_a": "mean",
        "persuasion_success_b": "mean"
    })
    condition_comparison.plot(kind="bar")
    plt.title("Persuasion Success by Speaker and Condition")
    plt.ylabel("Persuasion Rate")
    plt.xticks(rotation=45)
    plt.legend(["Model A Success", "Model B Success"])
    
    plt.tight_layout()
    plt.savefig(out_dir / "persuasion_analysis.png", dpi=150, bbox_inches="tight")
    
    print(f"\nResults saved to {out_dir}")
    print(f"Total sessions: {len(sessions)}")
    print(f"Overall persuasion rate: {df['any_persuasion'].mean():.3f}")
    print(f"Persuasion by condition:")
    for condition, rate in persuasion_by_condition.items():
        print(f"  {condition}: {rate:.3f}")

if __name__ == "__main__":
    main()
