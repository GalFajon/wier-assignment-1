import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EVAL_DATASET_PATH = "eval_datasets/evaluation_dataset.json"
ANSWERS_PATH = "eval_answers/ollama_chat-qwen3-14b_rag_one_shot_bge-m3_50_mmarco_3.json"

OUTPUT_PATH = "answer_quality_results.json"

OPENAI_JUDGE_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
TEMPERATURE = 0.0



ANSWER_QUALITY_JUDGE_PROMPT = """Si natančen ocenjevalec kakovosti odgovorov v slovenščini.

Dobil boš:
- vprašanje
- referenčni pričakovani odgovor
- seznam ključnih dejstev
- kandidatni odgovor modela

Tvoja naloga je oceniti, kako dobro kandidatni odgovor vsebinsko ustreza referenčnemu odgovoru in koliko ključnih dejstev pokrije.

Pomembna pravila:
- Ne zahtevaj dobesednega ujemanja. Parafraze so sprejemljive.
- Ne kaznuj drugačnega vrstnega reda informacij.
- Ne kaznuj manjših slogovnih razlik.
- Pri ključnih dejstvih bodi zmeren: če kandidatni odgovor zajame pomen dejstva, ga označi kot pokritega, tudi če uporablja drugačne besede.
- Če kandidatni odgovor dejstvo nakaže, vendar ga ne pove popolno, uporabi "partial".
- Če kandidatni odgovor dejstvo pravilno izrazi posredno, uporabi "implied".
- Če kandidatni odgovor ne vsebuje dejstva, uporabi "missing".
- Če kandidatni odgovor nasprotuje dejstvu, uporabi "contradicted".
- Dodatne pravilne informacije niso nujno napaka, če ne nasprotujejo referenčnemu odgovoru.
- Kaznuj izmišljene, napačne ali kontradiktorne informacije.
- Če kandidatni odgovor odgovori "ne vem" ali "ni mogoče ugotoviti", to ni halucinacija, vendar ključna dejstva praviloma niso pokrita.
- Ocenjuj vsebinsko pravilnost, ne kakovosti citiranja.
- Če so v odgovoru citati, jih ignoriraj pri oceni vsebine, razen če vplivajo na razumljivost.
- Vrni IZKLJUČNO veljaven JSON objekt.
- Ne oziraj se na citate oblike [ChunkID: <id>], ignoriraj jih, kot da niso v stavku.

Razlaga oznak za ključna dejstva:
- covered: dejstvo je jasno in pravilno pokrito
- implied: dejstvo je pravilno razvidno posredno, vendar ni izrečeno popolnoma neposredno
- partial: odgovor pokrije del dejstva, vendar manjka pomemben del
- missing: dejstva ni v odgovoru
- contradicted: odgovor dejstvu nasprotuje

Ocena overall_semantic_score:
- 1.0: odgovor je vsebinsko zelo dober in pokrije bistvo referenčnega odgovora
- 0.8: odgovor je večinoma pravilen, z manjšimi izpusti
- 0.6: odgovor je delno pravilen, vendar pomembno nepopoln
- 0.4: odgovor vsebuje nekaj relevantnih informacij, vendar je večinoma nepopoln ali nejasen
- 0.2: odgovor je večinoma napačen ali komaj relevanten
- 0.0: odgovor je napačen, nerelevanten ali popolnoma neodgovarjajoč

Vrni JSON v tej obliki:
{
  "overall_semantic_score": 0.0,
  "answer_addresses_question": true,
  "contains_major_error": false,
  "contains_contradiction": false,
  "contains_hallucination": false,
  "key_fact_judgments": [
    {
      "fact": "ključni fakt",
      "status": "covered | implied | partial | missing | contradicted",
      "score": 0.0,
      "reason": "kratek razlog v slovenščini"
    }
  ],
  "missing_facts": [],
  "incorrect_or_unsupported_claims": [],
  "short_reason": "kratek povzetek ocene v slovenščini"
}
"""


# =========================
# Loading helpers
# =========================

def load_eval_dataset(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    return raw.get("examples", [])


def load_answers_raw(answers_path: str) -> list[dict]:
    with open(answers_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    return raw.get("examples", raw.get("answers", []))


def build_answers_by_id_or_index(
    eval_dataset: list[dict],
    raw_answers: list[dict],
) -> dict[str, dict]:
    """
    Supports both formats:
    1. answer dicts contain eval_id
    2. answer dicts are stored in the same order as eval_dataset
    """
    answers_by_id = {}

    has_eval_ids = any(isinstance(item, dict) and item.get("eval_id") for item in raw_answers)

    if has_eval_ids:
        for item in raw_answers:
            if not isinstance(item, dict):
                continue

            eval_id = item.get("eval_id")
            if eval_id:
                answers_by_id[eval_id] = item

        return answers_by_id

    # Fallback: align by order
    for example, answer_item in zip(eval_dataset, raw_answers):
        eval_id = example.get("id")
        if eval_id:
            answers_by_id[eval_id] = answer_item

    return answers_by_id


def extract_answer_text(answer_item: Any) -> str:
    """
    Handles several possible answer formats:
    - {"answer": "..."}
    - {"model_answer": {"answer": "..."}}
    - plain string
    """
    if answer_item is None:
        return ""

    if isinstance(answer_item, str):
        return answer_item

    if not isinstance(answer_item, dict):
        return str(answer_item)

    if isinstance(answer_item.get("answer"), str):
        return answer_item["answer"]

    model_answer = answer_item.get("model_answer")
    if isinstance(model_answer, dict) and isinstance(model_answer.get("answer"), str):
        return model_answer["answer"]

    if isinstance(model_answer, str):
        return model_answer

    return ""


def normalize_key_facts(key_facts: list[Any]) -> list[dict]:
    normalized = []

    for fact in key_facts:
        if isinstance(fact, str):
            normalized.append({"fact": fact})
        elif isinstance(fact, dict) and fact.get("fact"):
            normalized.append({"fact": fact["fact"]})

    return normalized


# =========================
# OpenAI JSON call
# =========================

def call_openai_json(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_JUDGE_MODEL,
    temperature: float = TEMPERATURE,
    max_retries: int = MAX_RETRIES,
) -> dict | None:
    last_raw = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            last_raw = raw
            return json.loads(raw)

        except Exception as e:
            print(f"OpenAI judge error on attempt {attempt}/{max_retries}: {e}")
            if last_raw:
                print(f"Last raw response:\n{last_raw}")
            time.sleep(1.0 * attempt)

    return None


# =========================
# Scoring helpers
# =========================

FACT_STATUS_SCORES = {
    "covered": 1.0,
    "implied": 0.85,
    "partial": 0.5,
    "missing": 0.0,
    "contradicted": 0.0,
}


def build_answer_quality_prompt(example: dict, candidate_answer: str) -> str:
    key_facts = normalize_key_facts(example.get("key_facts", []))

    return f"""Vprašanje:
{example.get("query", "")}

Referenčni pričakovani odgovor:
{example.get("expected_answer", "")}

Ključna dejstva:
{json.dumps(key_facts, ensure_ascii=False, indent=2)}

Kandidatni odgovor modela:
{candidate_answer}
"""


def postprocess_judgment(judgment: dict, key_facts: list[dict]) -> dict:
    """
    Makes the LLM judgment safer:
    - clamps semantic score to [0, 1]
    - normalizes key fact statuses
    - computes key_fact_coverage_score from fact-level judgments
    """
    if not isinstance(judgment, dict):
        judgment = {}

    try:
        overall = float(judgment.get("overall_semantic_score", 0.0))
    except (TypeError, ValueError):
        overall = 0.0

    overall = max(0.0, min(1.0, overall))

    raw_fact_judgments = judgment.get("key_fact_judgments", [])
    if not isinstance(raw_fact_judgments, list):
        raw_fact_judgments = []

    # Map by fact text where possible.
    raw_by_fact = {}
    for item in raw_fact_judgments:
        if isinstance(item, dict) and item.get("fact"):
            raw_by_fact[item["fact"]] = item

    normalized_fact_judgments = []

    for fact_obj in key_facts:
        fact_text = fact_obj.get("fact", "")
        raw_item = raw_by_fact.get(fact_text)

        # Fallback: preserve order if exact fact-text matching fails.
        if raw_item is None and len(normalized_fact_judgments) < len(raw_fact_judgments):
            candidate = raw_fact_judgments[len(normalized_fact_judgments)]
            raw_item = candidate if isinstance(candidate, dict) else None

        if raw_item is None:
            status = "missing"
            score = 0.0
            reason = "Ocenjevalec ni vrnil presoje za to ključno dejstvo."
        else:
            status = str(raw_item.get("status", "missing")).strip().lower()
            if status not in FACT_STATUS_SCORES:
                status = "missing"

            # Prefer deterministic score from status to keep results comparable.
            score = FACT_STATUS_SCORES[status]
            reason = raw_item.get("reason", "")

        normalized_fact_judgments.append({
            "fact": fact_text,
            "status": status,
            "score": score,
            "reason": reason,
        })

    if normalized_fact_judgments:
        key_fact_coverage_score = sum(
            item["score"] for item in normalized_fact_judgments
        ) / len(normalized_fact_judgments)
    else:
        key_fact_coverage_score = None

    covered_count = sum(1 for item in normalized_fact_judgments if item["status"] == "covered")
    implied_count = sum(1 for item in normalized_fact_judgments if item["status"] == "implied")
    partial_count = sum(1 for item in normalized_fact_judgments if item["status"] == "partial")
    missing_count = sum(1 for item in normalized_fact_judgments if item["status"] == "missing")
    contradicted_count = sum(1 for item in normalized_fact_judgments if item["status"] == "contradicted")

    return {
        "overall_semantic_score": overall,
        "key_fact_coverage_score": key_fact_coverage_score,

        "answer_addresses_question": bool(judgment.get("answer_addresses_question", False)),
        "contains_major_error": bool(judgment.get("contains_major_error", False)),
        "contains_contradiction": bool(judgment.get("contains_contradiction", False)),
        "contains_hallucination": bool(judgment.get("contains_hallucination", False)),

        "covered_count": covered_count,
        "implied_count": implied_count,
        "partial_count": partial_count,
        "missing_count": missing_count,
        "contradicted_count": contradicted_count,
        "num_key_facts": len(normalized_fact_judgments),

        "key_fact_judgments": normalized_fact_judgments,

        "missing_facts": judgment.get("missing_facts", []),
        "incorrect_or_unsupported_claims": judgment.get("incorrect_or_unsupported_claims", []),
        "short_reason": judgment.get("short_reason", ""),
    }


def judge_answer_quality(example: dict, candidate_answer: str) -> dict:
    key_facts = normalize_key_facts(example.get("key_facts", []))
    prompt = build_answer_quality_prompt(example, candidate_answer)

    judgment = call_openai_json(
        system_prompt=ANSWER_QUALITY_JUDGE_PROMPT,
        user_prompt=prompt,
    )

    if judgment is None:
        judgment = {
            "overall_semantic_score": 0.0,
            "answer_addresses_question": False,
            "contains_major_error": True,
            "contains_contradiction": False,
            "contains_hallucination": False,
            "key_fact_judgments": [],
            "missing_facts": [],
            "incorrect_or_unsupported_claims": [],
            "short_reason": "OpenAI ocenjevanje ni uspelo.",
        }

    return postprocess_judgment(judgment, key_facts)


# =========================
# Main evaluation
# =========================

def run_eval(dataset_path: str, answers_path: str) -> tuple[list[dict], dict]:
    eval_dataset = load_eval_dataset(dataset_path)
    raw_answers = load_answers_raw(answers_path)
    answers_by_id = build_answers_by_id_or_index(eval_dataset, raw_answers)

    print(f"Loaded {len(eval_dataset)} evaluation examples")
    print(f"Loaded {len(raw_answers)} answer entries")
    print(f"Matched answers by id/order: {len(answers_by_id)}\n")

    results = []
    missing = []

    for i, example in enumerate(eval_dataset, start=1):
        eval_id = example.get("id")
        answer_item = answers_by_id.get(eval_id)

        if answer_item is None:
            missing.append(eval_id)
            continue

        candidate_answer = extract_answer_text(answer_item)

        print(f"Evaluating answer quality {i}/{len(eval_dataset)}: {eval_id}")

        quality = judge_answer_quality(
            example=example,
            candidate_answer=candidate_answer,
        )

        result = {
            "eval_id": eval_id,
            "query": example.get("query", ""),
            "expected_answer": example.get("expected_answer", ""),
            "candidate_answer": candidate_answer,

            **quality,
        }

        results.append(result)

        print(f"  overall_semantic_score : {quality['overall_semantic_score']:.2f}")

        if quality["key_fact_coverage_score"] is not None:
            print(f"  key_fact_coverage_score: {quality['key_fact_coverage_score']:.2f}")
        else:
            print("  key_fact_coverage_score: N/A")

        print(
            f"  facts: covered={quality['covered_count']} "
            f"implied={quality['implied_count']} "
            f"partial={quality['partial_count']} "
            f"missing={quality['missing_count']} "
            f"contradicted={quality['contradicted_count']}"
        )

        if quality["contains_hallucination"] or quality["contains_contradiction"]:
            print(f"  warning: hallucination={quality['contains_hallucination']} contradiction={quality['contains_contradiction']}")

        print()

    if missing:
        print(f"Missing answers for {len(missing)} examples: {missing}\n")

    aggregate = compute_aggregate_metrics(results)

    return results, aggregate


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def compute_aggregate_metrics(results: list[dict]) -> dict:
    n = len(results)

    if n == 0:
        return {
            "n": 0,
            "mean_overall_semantic_score": None,
            "mean_key_fact_coverage_score": None,
        }

    semantic_scores = [
        r["overall_semantic_score"]
        for r in results
        if r.get("overall_semantic_score") is not None
    ]

    key_fact_scores = [
        r["key_fact_coverage_score"]
        for r in results
        if r.get("key_fact_coverage_score") is not None
    ]

    hallucination_rate = sum(
        bool(r.get("contains_hallucination", False))
        for r in results
    ) / n

    contradiction_rate = sum(
        bool(r.get("contains_contradiction", False))
        for r in results
    ) / n

    major_error_rate = sum(
        bool(r.get("contains_major_error", False))
        for r in results
    ) / n

    addressed_rate = sum(
        bool(r.get("answer_addresses_question", False))
        for r in results
    ) / n

    total_key_facts = sum(r.get("num_key_facts", 0) for r in results)

    total_covered = sum(r.get("covered_count", 0) for r in results)
    total_implied = sum(r.get("implied_count", 0) for r in results)
    total_partial = sum(r.get("partial_count", 0) for r in results)
    total_missing = sum(r.get("missing_count", 0) for r in results)
    total_contradicted = sum(r.get("contradicted_count", 0) for r in results)

    if total_key_facts:
        strict_key_fact_coverage = total_covered / total_key_facts
        relaxed_key_fact_coverage = (
            total_covered
            + 0.85 * total_implied
            + 0.5 * total_partial
        ) / total_key_facts
    else:
        strict_key_fact_coverage = None
        relaxed_key_fact_coverage = None

    return {
        "n": n,

        # Main answer-quality metrics
        "mean_overall_semantic_score": mean(semantic_scores),
        "mean_key_fact_coverage_score": mean(key_fact_scores),

        # Useful diagnostic rates
        "answer_addressed_rate": addressed_rate,
        "hallucination_rate": hallucination_rate,
        "contradiction_rate": contradiction_rate,
        "major_error_rate": major_error_rate,

        # Corpus-level key-fact statistics
        "total_key_facts": total_key_facts,
        "total_covered": total_covered,
        "total_implied": total_implied,
        "total_partial": total_partial,
        "total_missing": total_missing,
        "total_contradicted": total_contradicted,
        "strict_key_fact_coverage": strict_key_fact_coverage,
        "relaxed_key_fact_coverage": relaxed_key_fact_coverage,
    }


def save_results(
    output_path: str,
    dataset_path: str,
    answers_path: str,
    results: list[dict],
    aggregate: dict,
) -> None:
    output = {
        "metadata": {
            "dataset_path": dataset_path,
            "answers_path": answers_path,
            "judge_model": OPENAI_JUDGE_MODEL,
            "temperature": TEMPERATURE,
        },
        "aggregate": aggregate,
        "examples": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved answer quality results to {output_path}")


def print_aggregate(aggregate: dict) -> None:
    print("=" * 60)
    print("ANSWER QUALITY AGGREGATE")
    print(f"n: {aggregate['n']}")

    if aggregate.get("mean_overall_semantic_score") is not None:
        print(f"mean_overall_semantic_score : {aggregate['mean_overall_semantic_score']:.3f}")
    else:
        print("mean_overall_semantic_score : N/A")

    if aggregate.get("mean_key_fact_coverage_score") is not None:
        print(f"mean_key_fact_coverage_score: {aggregate['mean_key_fact_coverage_score']:.3f}")
    else:
        print("mean_key_fact_coverage_score: N/A")

    if aggregate.get("strict_key_fact_coverage") is not None:
        print(f"strict_key_fact_coverage    : {aggregate['strict_key_fact_coverage']:.1%}")
    else:
        print("strict_key_fact_coverage    : N/A")

    if aggregate.get("relaxed_key_fact_coverage") is not None:
        print(f"relaxed_key_fact_coverage   : {aggregate['relaxed_key_fact_coverage']:.1%}")
    else:
        print("relaxed_key_fact_coverage   : N/A")

    print(f"answer_addressed_rate       : {aggregate['answer_addressed_rate']:.1%}")
    print(f"hallucination_rate          : {aggregate['hallucination_rate']:.1%}")
    print(f"contradiction_rate          : {aggregate['contradiction_rate']:.1%}")
    print(f"major_error_rate            : {aggregate['major_error_rate']:.1%}")

    print()
    print("KEY FACT COUNTS")
    print(f"total_key_facts      : {aggregate['total_key_facts']}")
    print(f"covered              : {aggregate['total_covered']}")
    print(f"implied              : {aggregate['total_implied']}")
    print(f"partial              : {aggregate['total_partial']}")
    print(f"missing              : {aggregate['total_missing']}")
    print(f"contradicted         : {aggregate['total_contradicted']}")


if __name__ == "__main__":
    results, aggregate = run_eval(EVAL_DATASET_PATH, ANSWERS_PATH)
    print_aggregate(aggregate)
    save_results(
        output_path=OUTPUT_PATH,
        dataset_path=EVAL_DATASET_PATH,
        answers_path=ANSWERS_PATH,
        results=results,
        aggregate=aggregate,
    )