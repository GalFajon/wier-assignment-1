import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EVAL_DATASET_PATH = "eval_datasets/evaluation_dataset.json"

ANSWERS_PATH = "eval_querying_styles/answers/ollama_chat-qwen3-14b_rag_zero_shot_bge-m3_50_bge_7.json"
OUTPUT_PATH = "ollama_chat-qwen3-14b_rag_zero_shot_bge-m3_50_bge_7_ANSWER_QUALITY.json"

OPENAI_JUDGE_MODEL = "gpt-4.1"
MAX_RETRIES = 3
TEMPERATURE = 0.0


ANSWER_QUALITY_JUDGE_PROMPT = """Si natančen ocenjevalec kakovosti odgovorov v slovenščini.

Dobil boš:
- vprašanje
- referenčni pričakovani odgovor
- seznam ključnih dejstev
- kandidatni odgovor modela

Tvoja naloga je oceniti, kako dobro kandidatni odgovor vsebinsko ustreza referenčnemu odgovoru in koliko ključnih dejstev pokrije.

Splošna pravila:
- Ne zahtevaj dobesednega ujemanja. Parafraze so sprejemljive.
- Ne kaznuj drugačnega vrstnega reda informacij.
- Ne kaznuj manjših slogovnih razlik.
- Ocenjuj vsebinsko pravilnost, ne kakovosti citiranja.
- Če so v odgovoru citati, jih ignoriraj pri oceni vsebine, razen če vplivajo na razumljivost.
- Ne oziraj se na citate oblike [ChunkID: <id>], ignoriraj jih, kot da niso v stavku.
- Dodatne pravilne informacije niso nujno napaka, če ne nasprotujejo referenčnemu odgovoru.
- Če kandidatni odgovor vsebuje pomembne dodatne trditve, ki niso podprte z referenčnim odgovorom ali ključnimi dejstvi, jih zabeleži v incorrect_or_unsupported_claims.
- Če kandidatni odgovor odgovori "ne vem", "ni mogoče ugotoviti" ali podobno, to ni halucinacija, vendar ključna dejstva praviloma niso pokrita.
- Vrni IZKLJUČNO veljaven JSON objekt.

Razlaga key_fact_judgments:
- Za vsako ključno dejstvo vrni en objekt v key_fact_judgments.
- Polje fact mora vsebovati isto ključno dejstvo, ki ga ocenjuješ.
- Polje status mora biti ena od vrednosti: covered, implied, partial, missing, contradicted.
- covered: dejstvo je jasno in pravilno pokrito.
- implied: dejstvo je pravilno razvidno posredno, vendar ni izrečeno popolnoma neposredno.
- partial: odgovor pokrije del dejstva, vendar manjka pomemben del.
- missing: dejstva ni v odgovoru.
- contradicted: odgovor neposredno nasprotuje dejstvu.
- Pri ključnih dejstvih bodi zmeren: če kandidatni odgovor zajame pomen dejstva, ga označi kot covered, tudi če uporablja drugačne besede.
- Če kandidatni odgovor dejstvo nakaže, vendar ga ne pove popolno, uporabi partial.
- Če kandidatni odgovor dejstvo pravilno izrazi posredno, uporabi implied.
- Če kandidatni odgovor ne vsebuje dejstva, uporabi missing.
- Če kandidatni odgovor nasprotuje dejstvu, uporabi contradicted.

Razlaga overall_semantic_score:
- overall_semantic_score naj oceni celotno uporabnost odgovora glede na vprašanje in referenčni pričakovani odgovor.
- Ne sme biti samo povprečje pokritosti ključnih dejstev.
- Upoštevaj neposrednost, jasnost, pravilnost, relevantnost, napačne dodatne trditve in ali odgovor dejansko odgovori na vprašanje.
- 1.0: odgovor je vsebinsko zelo dober, neposreden in pokrije bistvo referenčnega odgovora.
- 0.8: odgovor je večinoma pravilen, z manjšimi izpusti ali manjšimi nejasnostmi.
- 0.6: odgovor je delno pravilen, vendar pomembno nepopoln.
- 0.4: odgovor vsebuje nekaj relevantnih informacij, vendar je večinoma nepopoln, nejasen ali slabo usmerjen.
- 0.2: odgovor je večinoma napačen, komaj relevanten ali zelo pomanjkljiv.
- 0.0: odgovor je napačen, nerelevanten, prazen ali popolnoma neodgovarjajoč.

Razlaga answer_addresses_question:
- true, če kandidatni odgovor neposredno odgovarja na zastavljeno vprašanje, tudi če je nepopoln.
- false, če kandidatni odgovor ne odgovori na vprašanje, odgovori na drugo vprašanje, je prazen, ali se samo izogne odgovoru brez utemeljitve.
- Nepopoln odgovor je lahko answer_addresses_question=true, če se jasno nanaša na vprašanje.

Razlaga contains_major_error:
- true samo, če kandidatni odgovor vsebuje resno vsebinsko napako, ki bistveno poslabša pravilnost odgovora.
- Resna napaka pomeni, da odgovor odgovori na napačno vprašanje, napačno predstavi glavno dejstvo, vsebuje pomembno napačno ali izmišljeno trditev, vsebuje protislovje, ki bistveno spremeni pomen, ali je tako nepopoln, da uporabnika zavede glede glavnega odgovora.
- false, če odgovor samo izpusti manjša ali srednje pomembna dejstva.
- false, če je odgovor delno pravilen, vendar očitno nepopoln; to se že kaznuje z nižjim overall_semantic_score in missing/partial ključnimi dejstvi.

Razlaga contains_contradiction:
- true, če kandidatni odgovor neposredno nasprotuje referenčnemu odgovoru ali vsaj enemu ključnemu dejstvu.
- true tudi, če je vsaj eno ključno dejstvo označeno kot contradicted.
- false, če odgovor samo izpusti dejstvo ali ga pove nepopolno.

Razlaga contains_hallucination:
- true, če kandidatni odgovor vsebuje pomembno izmišljeno, nepreverljivo ali nepodprto trditev, ki ni razvidna iz referenčnega odgovora ali ključnih dejstev.
- true, če kandidatni odgovor dodaja konkretne številke, datume, imena, vzroke ali posledice, ki niso podprti z referenčnim odgovorom ali ključnimi dejstvi.
- false, če odgovor vsebuje samo parafraze, splošne formulacije ali manjše slogovne dodatke, ki ne spreminjajo pomena.
- false, če odgovor pravilno pove manj kot referenčni odgovor, vendar ne dodaja nepodprtih trditev.

Razlaga missing_facts:
- Vključi ključna dejstva, ki so označena kot missing.
- Lahko vključiš tudi ključna dejstva, ki so samo partial, če manjka pomemben del.
- Ne vključuj dejstev, ki so covered ali implied.

Razlaga incorrect_or_unsupported_claims:
- Vključi konkretne trditve iz kandidatnega odgovora, ki so napačne, kontradiktorne ali niso podprte z referenčnim odgovorom oziroma ključnimi dejstvi.
- Če takih trditev ni, vrni prazen seznam.
- Ne vključuj samo slogovnih pripomb.

Razlaga short_reason:
- Kratko povzemi glavni razlog za oceno v slovenščini.
- Omeni glavne pokrite informacije, glavne manjkajoče informacije ali večje napake, če obstajajo.

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
      "reason": "kratek razlog v slovenščini"
    }
  ],
  "missing_facts": [],
  "incorrect_or_unsupported_claims": [],
  "short_reason": "kratek povzetek ocene v slovenščini"
}
"""


FACT_STATUS_SCORES = {"covered": 1.0, "implied": 0.85, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}


def load_eval_dataset(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("examples", [])


def load_answers_raw(answers_path: str) -> list[dict]:
    with open(answers_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("examples", raw.get("answers", []))


def build_answers_by_id_or_index(eval_dataset: list[dict], raw_answers: list[dict]) -> dict[str, dict]:
    answers_by_id = {}
    has_eval_ids = any(isinstance(item, dict) and item.get("eval_id") for item in raw_answers)

    if has_eval_ids:
        for item in raw_answers:
            if isinstance(item, dict) and item.get("eval_id"):
                answers_by_id[str(item["eval_id"])] = item
        return answers_by_id

    for example, answer_item in zip(eval_dataset, raw_answers):
        eval_id = example.get("id")
        if eval_id:
            answers_by_id[str(eval_id)] = answer_item

    return answers_by_id


def extract_answer_text(answer_item: Any) -> str:
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
            normalized.append({"fact": str(fact["fact"])})
    return normalized


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return [str(value)]


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def call_openai_json(system_prompt: str, user_prompt: str, model: str = OPENAI_JUDGE_MODEL, temperature: float = TEMPERATURE, max_retries: int = MAX_RETRIES) -> dict | None:
    last_raw = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
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


def compute_answer_quality_score(result: dict) -> float:
    coverage = clamp01(result.get("key_fact_coverage_score") if result.get("key_fact_coverage_score") is not None else 0.0)
    semantic = clamp01(result.get("overall_semantic_score", 0.0))
    score = 0.65 * coverage + 0.35 * semantic

    if result.get("contains_contradiction", False):
        score *= 0.5
    if result.get("contains_major_error", False):
        score *= 0.5
    if not result.get("answer_addresses_question", False):
        score *= 0.5

    return clamp01(score)


def postprocess_judgment(judgment: dict, key_facts: list[dict]) -> dict:
    if not isinstance(judgment, dict):
        judgment = {}

    overall = clamp01(judgment.get("overall_semantic_score", 0.0))
    raw_fact_judgments = judgment.get("key_fact_judgments", [])
    if not isinstance(raw_fact_judgments, list):
        raw_fact_judgments = []

    raw_by_fact = {str(item["fact"]): item for item in raw_fact_judgments if isinstance(item, dict) and item.get("fact")}
    normalized_fact_judgments = []

    for fact_obj in key_facts:
        fact_text = fact_obj.get("fact", "")
        raw_item = raw_by_fact.get(fact_text)

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
            score = FACT_STATUS_SCORES[status]
            reason = str(raw_item.get("reason", ""))

        normalized_fact_judgments.append({"fact": fact_text, "status": status, "score": score, "reason": reason})

    key_fact_coverage_score = sum(item["score"] for item in normalized_fact_judgments) / len(normalized_fact_judgments) if normalized_fact_judgments else None
    covered_count = sum(1 for item in normalized_fact_judgments if item["status"] == "covered")
    implied_count = sum(1 for item in normalized_fact_judgments if item["status"] == "implied")
    partial_count = sum(1 for item in normalized_fact_judgments if item["status"] == "partial")
    missing_count = sum(1 for item in normalized_fact_judgments if item["status"] == "missing")
    contradicted_count = sum(1 for item in normalized_fact_judgments if item["status"] == "contradicted")

    result = {
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
        "missing_facts": normalize_string_list(judgment.get("missing_facts", [])),
        "incorrect_or_unsupported_claims": normalize_string_list(judgment.get("incorrect_or_unsupported_claims", [])),
        "short_reason": str(judgment.get("short_reason", "")),
        "raw_judgment": judgment,
    }

    result["answer_quality_score"] = compute_answer_quality_score(result)
    return result


def empty_or_missing_answer_quality(example: dict, reason: str) -> dict:
    key_facts = normalize_key_facts(example.get("key_facts", []))
    key_fact_judgments = [{"fact": fact.get("fact", ""), "status": "missing", "score": 0.0, "reason": reason} for fact in key_facts]

    result = {
        "overall_semantic_score": 0.0,
        "key_fact_coverage_score": 0.0,
        "answer_addresses_question": False,
        "contains_major_error": True,
        "contains_contradiction": False,
        "contains_hallucination": False,
        "covered_count": 0,
        "implied_count": 0,
        "partial_count": 0,
        "missing_count": len(key_facts),
        "contradicted_count": 0,
        "num_key_facts": len(key_facts),
        "key_fact_judgments": key_fact_judgments,
        "missing_facts": [fact.get("fact", "") for fact in key_facts],
        "incorrect_or_unsupported_claims": [],
        "short_reason": reason,
        "raw_judgment": None,
    }

    result["answer_quality_score"] = compute_answer_quality_score(result)
    return result


def judge_answer_quality(example: dict, candidate_answer: str) -> dict:
    key_facts = normalize_key_facts(example.get("key_facts", []))

    if not candidate_answer or not candidate_answer.strip():
        return empty_or_missing_answer_quality(example=example, reason="Kandidatni odgovor je prazen.")

    judgment = call_openai_json(system_prompt=ANSWER_QUALITY_JUDGE_PROMPT, user_prompt=build_answer_quality_prompt(example, candidate_answer))

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
        eval_id = str(example.get("id", f"missing_id_{i}"))
        answer_item = answers_by_id.get(eval_id)

        print(f"Evaluating answer quality {i}/{len(eval_dataset)}: {eval_id}")

        if answer_item is None:
            missing.append(eval_id)
            candidate_answer = ""
            quality = empty_or_missing_answer_quality(example=example, reason="Manjka kandidatni odgovor.")
        else:
            candidate_answer = extract_answer_text(answer_item)
            quality = judge_answer_quality(example=example, candidate_answer=candidate_answer)

        result = {
            "eval_id": eval_id,
            "query": example.get("query", ""),
            "expected_answer": example.get("expected_answer", ""),
            "candidate_answer": candidate_answer,
            **quality,
        }

        results.append(result)

        print(f"  overall_semantic_score : {quality['overall_semantic_score']:.2f}")
        print(f"  key_fact_coverage_score: {quality['key_fact_coverage_score']:.2f}" if quality["key_fact_coverage_score"] is not None else "  key_fact_coverage_score: N/A")
        print(f"  answer_quality_score   : {quality['answer_quality_score']:.2f}")
        print(f"  facts: covered={quality['covered_count']} implied={quality['implied_count']} partial={quality['partial_count']} missing={quality['missing_count']} contradicted={quality['contradicted_count']}")

        if quality["contains_hallucination"] or quality["contains_contradiction"]:
            print(f"  warning: hallucination={quality['contains_hallucination']} contradiction={quality['contains_contradiction']}")

        print()

    if missing:
        print(f"Missing answers for {len(missing)} examples: {missing}\n")

    return results, compute_aggregate_metrics(results)


def compute_aggregate_metrics(results: list[dict]) -> dict:
    n = len(results)

    if n == 0:
        return {"n": 0, "mean_overall_semantic_score": None, "mean_key_fact_coverage_score": None, "mean_answer_quality_score": None}

    semantic_scores = [r["overall_semantic_score"] for r in results if r.get("overall_semantic_score") is not None]
    key_fact_scores = [r["key_fact_coverage_score"] for r in results if r.get("key_fact_coverage_score") is not None]
    answer_quality_scores = [r["answer_quality_score"] for r in results if r.get("answer_quality_score") is not None]

    hallucination_rate = sum(bool(r.get("contains_hallucination", False)) for r in results) / n
    contradiction_rate = sum(bool(r.get("contains_contradiction", False)) for r in results) / n
    major_error_rate = sum(bool(r.get("contains_major_error", False)) for r in results) / n
    addressed_rate = sum(bool(r.get("answer_addresses_question", False)) for r in results) / n

    total_key_facts = sum(r.get("num_key_facts", 0) for r in results)
    total_covered = sum(r.get("covered_count", 0) for r in results)
    total_implied = sum(r.get("implied_count", 0) for r in results)
    total_partial = sum(r.get("partial_count", 0) for r in results)
    total_missing = sum(r.get("missing_count", 0) for r in results)
    total_contradicted = sum(r.get("contradicted_count", 0) for r in results)
    strict_key_fact_coverage = total_covered / total_key_facts if total_key_facts else None

    return {
        "n": n,
        "mean_overall_semantic_score": mean(semantic_scores),
        "mean_key_fact_coverage_score": mean(key_fact_scores),
        "mean_answer_quality_score": mean(answer_quality_scores),
        "answer_addressed_rate": addressed_rate,
        "hallucination_rate": hallucination_rate,
        "contradiction_rate": contradiction_rate,
        "major_error_rate": major_error_rate,
        "total_key_facts": total_key_facts,
        "total_covered": total_covered,
        "total_implied": total_implied,
        "total_partial": total_partial,
        "total_missing": total_missing,
        "total_contradicted": total_contradicted,
        "strict_key_fact_coverage": strict_key_fact_coverage,
    }


def save_results(output_path: str, dataset_path: str, answers_path: str, results: list[dict], aggregate: dict) -> None:
    output = {
        "metadata": {
            "dataset_path": dataset_path,
            "answers_path": answers_path,
            "judge_model": OPENAI_JUDGE_MODEL,
            "temperature": TEMPERATURE,
            "max_retries": MAX_RETRIES,
            "scoring": {
                "primary_metrics": ["mean_overall_semantic_score", "mean_key_fact_coverage_score"],
                "ranking_metric": "mean_answer_quality_score",
                "answer_quality_score_formula": "0.65 * key_fact_coverage_score + 0.35 * overall_semantic_score, with penalties for contradiction, major_error, and not addressing question",
            },
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
    print(f"mean_overall_semantic_score : {aggregate['mean_overall_semantic_score']:.3f}" if aggregate.get("mean_overall_semantic_score") is not None else "mean_overall_semantic_score : N/A")
    print(f"mean_key_fact_coverage_score: {aggregate['mean_key_fact_coverage_score']:.3f}" if aggregate.get("mean_key_fact_coverage_score") is not None else "mean_key_fact_coverage_score: N/A")
    print(f"mean_answer_quality_score   : {aggregate['mean_answer_quality_score']:.3f}" if aggregate.get("mean_answer_quality_score") is not None else "mean_answer_quality_score   : N/A")
    print(f"strict_key_fact_coverage    : {aggregate['strict_key_fact_coverage']:.1%}" if aggregate.get("strict_key_fact_coverage") is not None else "strict_key_fact_coverage    : N/A")
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
    save_results(output_path=OUTPUT_PATH, dataset_path=EVAL_DATASET_PATH, answers_path=ANSWERS_PATH, results=results, aggregate=aggregate)