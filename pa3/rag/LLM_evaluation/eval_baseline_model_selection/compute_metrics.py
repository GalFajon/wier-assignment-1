#!/usr/bin/env python3

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class ModelMetrics:
    model_name: str
    n_examples: int
    mean_key_fact_coverage: float
    mean_semantic_score: float
    mean_composite_quality: float
    hallucination_rate: float
    contradiction_rate: float
    major_error_rate: float
    question_address_rate: float
    key_fact_breakdown: Dict[str, int]
    
    def __str__(self) -> str:
        """Format metrics for display."""
        breakdown_str = (
            f"covered={self.key_fact_breakdown['covered']}, "
            f"implied={self.key_fact_breakdown['implied']}, "
            f"partial={self.key_fact_breakdown['partial']}, "
            f"missing={self.key_fact_breakdown['missing']}, "
            f"contradicted={self.key_fact_breakdown['contradicted']}"
        )
        return (
            f"{self.model_name}:\n"
            f"  N examples: {self.n_examples}\n"
            f"  Mean key-fact coverage: {self.mean_key_fact_coverage:.3f}\n"
            f"  Mean semantic score: {self.mean_semantic_score:.2f}\n"
            f"  Mean composite quality: {self.mean_composite_quality:.3f}\n"
            f"  Key-fact breakdown: {breakdown_str}\n"
            f"  Hallucination rate: {self.hallucination_rate:.0%}\n"
            f"  Contradiction rate: {self.contradiction_rate:.2%}\n"
            f"  Major-error rate: {self.major_error_rate:.0%}\n"
            f"  Question-address rate: {self.question_address_rate:.0%}"
        )


def load_json_file(filepath: Path) -> dict:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return None


def extract_model_name(filename: str) -> str:
    import re
    match = re.search(r'qwen3?-(\d+b)', filename.lower())
    if match:
        size = match.group(1).upper()
        return f"Qwen {size}"
    return filename


def compute_metrics(data: dict) -> ModelMetrics:
    if not data:
        return None
    
    metadata = data.get('metadata', {})
    aggregate = data.get('aggregate', {})
    
    model_name = extract_model_name(metadata.get('answers_path', 'Unknown'))
    
    metrics = ModelMetrics(
        model_name=model_name,
        n_examples=aggregate.get('n', 0),
        mean_key_fact_coverage=aggregate.get('mean_key_fact_coverage_score', 0),
        mean_semantic_score=aggregate.get('mean_overall_semantic_score', 0),
        mean_composite_quality=aggregate.get('mean_answer_quality_score', 0),
        hallucination_rate=aggregate.get('hallucination_rate', 0),
        contradiction_rate=aggregate.get('contradiction_rate', 0),
        major_error_rate=aggregate.get('major_error_rate', 0),
        question_address_rate=aggregate.get('answer_addressed_rate', 0),
        key_fact_breakdown={
            'covered': aggregate.get('total_covered', 0),
            'implied': aggregate.get('total_implied', 0),
            'partial': aggregate.get('total_partial', 0),
            'missing': aggregate.get('total_missing', 0),
            'contradicted': aggregate.get('total_contradicted', 0),
        }
    )
    
    return metrics


def main():
    dirpath = Path(__file__).parent / "quality_evaluations"
    
    if not dirpath.exists():
        print(f"Error: Directory not found: {dirpath}")
        return

    json_files = sorted(dirpath.glob("*_ANSWER_QUALITY.json"))
    
    if not json_files:
        print(f"No answer quality evaluation files found in {dirpath}")
        return

    print("=" * 80)
    print()
    
    all_metrics = []
    for json_file in json_files:
        data = load_json_file(json_file)
        if data:
            metrics = compute_metrics(data)
            if metrics:
                all_metrics.append(metrics)
                print(metrics)
                print()

    print("=" * 80)
    print()
    print(f"{'Model':<12} {'Coverage':<10} {'Semantic':<10} {'Composite':<10} {'Hall%':<7} {'Contra%':<8} {'Error%':<7} {'Addr%':<6}")
    print("-" * 80)
    for m in all_metrics:
        print(
            f"{m.model_name:<12} {m.mean_key_fact_coverage:.3f}     "
            f"{m.mean_semantic_score:.2f}      {m.mean_composite_quality:.3f}     "
            f"{m.hallucination_rate*100:5.0f}  {m.contradiction_rate*100:6.2f}  "
            f"{m.major_error_rate*100:5.0f}  {m.question_address_rate*100:4.0f}"
        )
    
    print()

main()
