import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OPEN_AI_MODEL = "gpt-4.1-mini"

INPUT_FILE = "sampled_pages.json"
OUTPUT_FILE = "generated_queries.json"

MAX_PAGES = None # none is all pages
MAX_WINDOWS_PER_PAGE = 1 # none is all windows per page

TEMPERATURE = 0.2
MAX_RETRIES = 3

BAD_QUERY_PATTERNS = [
    "kakšen je pomen",
    "kakšen je vpliv",
    "kako je to povezano",
    "zakaj je to pomembno",
    "kaj lahko sklepamo",
    "povzemi",
    "kakšno je stanje",
    "kakšen je kontekst",
    "razpravljaj",
    "analiziraj",
] # prevents queries to be overfitted to chunk

SYSTEM_PROMPT = """Si pomočnik za generiranje evalvacijskega nabora podatkov za ocenjevanje RAG sistema.

Dobil boš:
1. IZBRANO IZVORNO OKNO: enega ali več besedilnih odsekov, ki so bili izbrani kot predvideni dokazni vir.
2. DRUGE ODSEKE Z ISTE SPLETNE STRANI: dodatne odseke, ki lahko nudijo kontekst.

Tvoja naloga:
- Ustvari ENO specifično dejstveno evalvacijsko vprašanje v slovenščini.
- Vprašanje mora biti odgovorljivo z uporabo IZBRANEGA IZVORNEGA OKNA.
- Če ima izbrano izvorno okno več odsekov, poskusi oblikovati vprašanje tako, da vsak izbrani odsek prispeva vsaj eno ločeno dejstveno trditev.
- Vprašanje ne sme zahtevati zunanjega znanja.
- Vprašanje ne sme biti široko vprašanje za povzemanje ali mnenjsko vprašanje.
- Prednost daj vprašanjem o konkretnih dogodkih, akterjih, datumih, številkah, lokacijah, trditvah, citiranih izjavah ali pripisovanju trditev virom.
- Pričakovani odgovor mora biti jedrnat, v slovenščini in v celoti podprt z izbranim izvornim oknom.
- Ključna dejstva morajo biti atomske trditve iz pričakovanega odgovora.
- Vsako ključno dejstvo mora navesti ID-je izbranih odsekov, ki ga podpirajo.
- Kandidatni podporni odseki naj vključujejo samo druge odseke, za katere se zdi, da neposredno podpirajo isti odgovor ali ista ključna dejstva.
- Ne vključuj odsekov, ki so samo tematsko povezani.

Pomembno:
- Če izbrano izvorno okno ni primerno za dobro vprašanje, vrni usable=false.
- Ne izmišljaj dejstev.
- Ne uporabljaj informacij, ki niso prisotne v odsekih.
- Vprašanje, pričakovani odgovor, ključna dejstva in razlogi morajo biti v slovenščini.
- Imena JSON ključev in dovoljene oznake v poljih morajo ostati točno takšne, kot so navedene spodaj.
- Vrni IZKLJUČNO veljaven JSON objekt. Brez markdowna, brez dodatne razlage.

Vrni točno to JSON strukturo:
{
  "usable": true,
  "query": "specifično vprašanje v slovenščini",
  "expected_answer": "jedrnat odgovor v slovenščini, podprt z izbranim izvornim oknom",
  "required_chunk_ids": [123],
  "key_facts": [
    {
      "fact": "atomska dejstvena trditev v slovenščini",
      "supported_by": [123]
    }
  ],
  "candidate_supporting_chunk_ids": [124, 125],
  "question_type": "factual | temporal | attribution | number | comparison | multi_hop",
  "difficulty": "easy | medium | hard"
}

Če izbrano okno ni primerno, vrni:
{
  "usable": false,
  "reason": "kratek razlog v slovenščini"
}
"""


ANSWERABILITY_VERIFIER_PROMPT = """Preverjaš veljavnost generiranega evalvacijskega primera za RAG sistem.

Dobil boš:
- izbrano izvorno okno
- generirano vprašanje
- generiran pričakovani odgovor
- generirana ključna dejstva

Tvoja naloga:
Preveri, ali je pričakovani odgovor v celoti podprt z izbranim izvornim oknom.

Pravila:
- Odgovor ne sme zahtevati zunanjega znanja.
- Vsaka dejstvena trditev v pričakovanem odgovoru mora biti podprta z izbranim izvornim oknom.
- Ključna dejstva morajo biti atomska.
- Vsako ključno dejstvo mora biti podprto z vsaj enim izbranim izvornim odsekom.
- Pri izvornih oknih z več odseki določi, kateri izbrani odseki so dejansko potrebni.
- Razlog mora biti v slovenščini.
- Imena JSON ključev in vrednosti true/false morajo ostati točno takšne, kot so navedene spodaj.
- Vrni IZKLJUČNO veljaven JSON objekt.

Vrni:
{
  "valid": true,
  "required_chunk_ids": [123],
  "unnecessary_selected_chunk_ids": [],
  "reason": "kratek razlog v slovenščini"
}

Če primer ni veljaven, vrni:
{
  "valid": false,
  "required_chunk_ids": [],
  "unnecessary_selected_chunk_ids": [],
  "reason": "kratek razlog v slovenščini"
}
"""


SUPPORT_VERIFIER_PROMPT = """Si preverjevalec podpornih oznak za RAG evalvacijski nabor.

Dobil boš:
- vprašanje
- pričakovani odgovor
- ključna dejstva
- en kandidatni odsek

Tvoja naloga:
Določi, kakšno vrsto podpore kandidatni odsek nudi za vprašanje, pričakovani odgovor ali morebiten pravilen odgovor.

Oznake:
- supports_expected_answer: odsek neposredno podpira pričakovani odgovor ali vsaj eno ključno dejstvo
- partial_support: odsek vsebuje relevantno informacijo za odgovor na vprašanje, vendar ne podpira dovolj pričakovanega odgovora ali ključnih dejstev
- contradicts: odsek nasprotuje pričakovanemu odgovoru ali ključnim dejstvom
- related_but_not_answering: odsek je tematsko povezan, vendar ne pomaga odgovoriti na vprašanje
- unrelated: odsek ni relevanten za vprašanje

Pomembno:
- Če odsek vsebuje dodatno dejstvo, ki bi lahko bilo del pravilnega odgovora na vprašanje, uporabi partial_support, ne hard negative.
- Če odsek podpira katero koli ključno dejstvo, uporabi supports_expected_answer.
- Če je odsek samo o isti splošni temi, vendar ne pomaga odgovoriti na vprašanje, uporabi related_but_not_answering.
- Razlog mora biti v slovenščini.
- Vrni IZKLJUČNO veljaven JSON objekt.

Vrni:
{
  "verdict": "supports_expected_answer | partial_support | contradicts | related_but_not_answering | unrelated",
  "supported_facts": ["sem kopiraj natančna podprta ključna dejstva"],
  "additional_relevant_information": ["sem napiši dodatne relevantne informacije, če obstajajo"],
  "reason": "kratek razlog v slovenščini"
}
"""




def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = TEMPERATURE,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any] | None:
    """
    Calls the LLM and parses a JSON object response.
    Returns None if all retries fail.
    """
    last_raw = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
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
            print(f"LLM/JSON error on attempt {attempt}/{max_retries}: {e}")
            if last_raw:
                print(f"Last raw response:\n{last_raw}")
            time.sleep(1.0 * attempt)

    return None


def normalize_chunk_id(chunk_id: Any) -> str:
    return str(chunk_id)


def make_chunk_map(page: dict) -> dict[str, dict]:
    return {
        normalize_chunk_id(c["chunk_id"]): c
        for c in page["chunks"]
    }


def format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[chunk_id={c['chunk_id']}]\n{c['text']}"
        for c in chunks
    )


def get_source_window_chunks(page: dict, source_window: dict) -> list[dict]:
    chunk_map = make_chunk_map(page)
    chunks = []

    for cid in source_window["chunk_ids"]:
        key = normalize_chunk_id(cid)
        if key in chunk_map:
            chunks.append(chunk_map[key])

    return chunks


def get_other_chunks(page: dict, source_window: dict) -> list[dict]:
    source_ids = {
        normalize_chunk_id(cid)
        for cid in source_window["chunk_ids"]
    }

    return [
        c for c in page["chunks"]
        if normalize_chunk_id(c["chunk_id"]) not in source_ids
    ]


def validate_chunk_ids(ids: list[Any], chunk_map: dict[str, dict]) -> list[str]:
    valid = []

    for cid in ids:
        key = normalize_chunk_id(cid)
        if key in chunk_map:
            valid.append(key)

    return valid


def verify_numbers(
    obj: dict,
    chunk_map: dict[str, dict],
    fields: list[str] | None = None,
) -> tuple[bool, dict]:
    """
    Algorithmically verifies that all chunk IDs returned by the LLM exist
    in the current page's chunk_map.

    This checks:
    - top-level list fields such as required_chunk_ids
    - top-level list fields such as candidate_supporting_chunk_ids
    - key_facts[*].supported_by

    It does not call the LLM. It only checks ID validity.

    Returns:
    - is_valid: bool
    - report: dict with invalid IDs and normalized valid IDs
    """
    if fields is None:
        fields = [
            "required_chunk_ids",
            "candidate_supporting_chunk_ids",
            "unnecessary_selected_chunk_ids",
        ]

    valid_chunk_ids = set(chunk_map.keys())

    report = {
        "valid_chunk_ids_on_page": sorted(
            valid_chunk_ids,
            key=lambda x: int(x) if x.isdigit() else x,
        ),
        "checked_fields": fields,
        "invalid_ids": {},
        "valid_ids": {},
        "key_fact_invalid_ids": [],
    }

    all_valid = True

    for field in fields:
        raw_ids = obj.get(field, [])

        if raw_ids is None:
            raw_ids = []

        if not isinstance(raw_ids, list):
            all_valid = False
            report["invalid_ids"][field] = [{
                "value": raw_ids,
                "reason": "field_is_not_a_list",
            }]
            continue

        normalized_ids = [normalize_chunk_id(cid) for cid in raw_ids]
        valid_ids = [cid for cid in normalized_ids if cid in valid_chunk_ids]
        invalid_ids = [cid for cid in normalized_ids if cid not in valid_chunk_ids]

        report["valid_ids"][field] = valid_ids

        if invalid_ids:
            all_valid = False
            report["invalid_ids"][field] = invalid_ids

    key_facts = obj.get("key_facts", [])

    if key_facts is None:
        key_facts = []

    if not isinstance(key_facts, list):
        all_valid = False
        report["invalid_ids"]["key_facts"] = [{
            "value": key_facts,
            "reason": "key_facts_is_not_a_list",
        }]
        return all_valid, report

    for i, fact in enumerate(key_facts):
        if not isinstance(fact, dict):
            all_valid = False
            report["key_fact_invalid_ids"].append({
                "key_fact_index": i,
                "value": fact,
                "reason": "key_fact_is_not_an_object",
            })
            continue

        supported_by = fact.get("supported_by", [])

        if supported_by is None:
            supported_by = []

        if not isinstance(supported_by, list):
            all_valid = False
            report["key_fact_invalid_ids"].append({
                "key_fact_index": i,
                "field": "supported_by",
                "value": supported_by,
                "reason": "supported_by_is_not_a_list",
            })
            continue

        normalized_supported_by = [
            normalize_chunk_id(cid)
            for cid in supported_by
        ]

        invalid_supported_by = [
            cid for cid in normalized_supported_by
            if cid not in valid_chunk_ids
        ]

        if invalid_supported_by:
            all_valid = False
            report["key_fact_invalid_ids"].append({
                "key_fact_index": i,
                "fact": fact.get("fact"),
                "field": "supported_by",
                "invalid_ids": invalid_supported_by,
            })

    return all_valid, report


def is_bad_query(query: str) -> bool:
    q = query.lower().strip()
    return any(pattern in q for pattern in BAD_QUERY_PATTERNS)


def build_generation_prompt(page: dict, source_window: dict) -> str:
    source_chunks = get_source_window_chunks(page, source_window)
    other_chunks = get_other_chunks(page, source_window)

    return f"""Metapodatki strani:
page_id: {page["page_id"]}
url: {page["url"]}

Metapodatki izbranega izvornega okna:
window_id: {source_window.get("window_id")}
window_size: {source_window.get("window_size")}
position_bucket: {source_window.get("position_bucket")}
selected_chunk_ids: {source_window.get("chunk_ids")}

IZBRANO IZVORNO OKNO:
{format_chunks(source_chunks)}

DRUGI ODSEKI Z ISTE SPLETNE STRANI:
{format_chunks(other_chunks)}
"""


def build_answerability_verifier_prompt(
    page: dict,
    source_window: dict,
    generated: dict,
) -> str:
    source_chunks = get_source_window_chunks(page, source_window)

    return f"""Izbrano izvorno okno:
{format_chunks(source_chunks)}

Generirano vprašanje:
{generated.get("query")}

Generiran pričakovani odgovor:
{generated.get("expected_answer")}

Generirana ključna dejstva:
{json.dumps(generated.get("key_facts", []), ensure_ascii=False, indent=2)}

ID-ji izbranih izvornih odsekov:
{source_window.get("chunk_ids")}
"""


def build_support_verifier_prompt(
    question: str,
    expected_answer: str,
    key_facts: list[dict],
    candidate_chunk: dict,
) -> str:
    return f"""Vprašanje:
{question}

Pričakovani odgovor:
{expected_answer}

Ključna dejstva:
{json.dumps(key_facts, ensure_ascii=False, indent=2)}

Kandidatni odsek:
[chunk_id={candidate_chunk["chunk_id"]}]
{candidate_chunk["text"]}
"""


def verify_answerability(page: dict, source_window: dict, generated: dict) -> dict | None:
    prompt = build_answerability_verifier_prompt(page, source_window, generated)
    return call_llm_json(
        system_prompt=ANSWERABILITY_VERIFIER_PROMPT,
        user_prompt=prompt,
        temperature=0.0,
    )


def verify_candidate_supports(
    page: dict,
    generated: dict,
    candidate_ids: list[Any],
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """
    Returns:
    - acceptable_support_ids
    - partial_support_ids
    - hard_negative_ids
    - detailed judgments
    """
    chunk_map = make_chunk_map(page)

    ids_valid, ids_report = verify_numbers(
        {"candidate_supporting_chunk_ids": candidate_ids},
        chunk_map,
        fields=["candidate_supporting_chunk_ids"],
    )

    if not ids_valid:
        print("Invalid candidate_supporting_chunk_ids detected:")
        print(json.dumps(ids_report, ensure_ascii=False, indent=2))

    valid_candidate_ids = validate_chunk_ids(candidate_ids, chunk_map)

    acceptable = []
    partial_supports = []
    hard_negatives = []
    judgments = []

    for cid in valid_candidate_ids:
        candidate_chunk = chunk_map[cid]

        prompt = build_support_verifier_prompt(
            question=generated["query"],
            expected_answer=generated["expected_answer"],
            key_facts=generated.get("key_facts", []),
            candidate_chunk=candidate_chunk,
        )

        judgment = call_llm_json(
            system_prompt=SUPPORT_VERIFIER_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
        )

        if not judgment:
            continue

        verdict = judgment.get("verdict")

        judgments.append({
            "chunk_id": cid,
            "verdict": verdict,
            "supported_facts": judgment.get("supported_facts", []),
            "additional_relevant_information": judgment.get("additional_relevant_information", []),
            "reason": judgment.get("reason", ""),
        })

        if verdict == "supports_expected_answer":
            acceptable.append(cid)

        elif verdict == "partial_support":
            partial_supports.append(cid)

        elif verdict in {"contradicts", "unrelated"}:
            hard_negatives.append(cid)

        elif verdict == "related_but_not_answering":
            partial_supports.append(cid)

    return acceptable, partial_supports, hard_negatives, judgments


def generate_query_for_source_window(page: dict, source_window: dict) -> dict | None:
    chunk_map = make_chunk_map(page)
    source_chunk_ids = [
        normalize_chunk_id(cid)
        for cid in source_window["chunk_ids"]
    ]

    # Safety: source window must point to real chunks.
    if any(cid not in chunk_map for cid in source_chunk_ids):
        print(f"Invalid source window chunk IDs on page {page['page_id']}: {source_chunk_ids}")
        return None

    generation_prompt = build_generation_prompt(page, source_window)

    generated = call_llm_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=generation_prompt,
        temperature=TEMPERATURE,
    )

    if not generated:
        return None

    if generated.get("usable") is False:
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "generation",
            "reason": generated.get("reason", "No reason provided"),
        }

    generated_ids_valid, generated_ids_report = verify_numbers(
        generated,
        chunk_map,
        fields=[
            "required_chunk_ids",
            "candidate_supporting_chunk_ids",
        ],
    )

    if not generated_ids_valid:
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "generated_id_validation",
            "reason": "Generated response contains chunk IDs that do not exist on this page",
            "raw_generated": generated,
            "id_validation_report": generated_ids_report,
        }

    required = validate_chunk_ids(generated.get("required_chunk_ids", []), chunk_map)

    if not required:
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "id_validation",
            "reason": "No valid required_chunk_ids returned",
            "raw_generated": generated,
        }

    query = generated.get("query", "").strip()
    expected_answer = generated.get("expected_answer", "").strip()

    if not query or not expected_answer:
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "content_validation",
            "reason": "Missing query or expected_answer",
            "raw_generated": generated,
        }

    if is_bad_query(query):
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "query_quality",
            "reason": "Query matched a broad/vague bad-query pattern",
            "raw_generated": generated,
        }

    answerability = verify_answerability(page, source_window, generated)

    if not answerability or answerability.get("valid") is not True:
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "answerability_verification",
            "reason": None if not answerability else answerability.get("reason"),
            "raw_generated": generated,
            "answerability_judgment": answerability,
        }

    answerability_ids_valid, answerability_ids_report = verify_numbers(
        answerability,
        chunk_map,
        fields=[
            "required_chunk_ids",
            "unnecessary_selected_chunk_ids",
        ],
    )

    if not answerability_ids_valid:
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "answerability_id_validation",
            "reason": "Answerability verifier returned chunk IDs that do not exist on this page",
            "raw_generated": generated,
            "answerability_judgment": answerability,
            "id_validation_report": answerability_ids_report,
        }

    verified_required = validate_chunk_ids(
        answerability.get("required_chunk_ids", []),
        chunk_map,
    )

    if not verified_required:
        return {
            "page_id": page["page_id"],
            "url": page["url"],
            "source_window": source_window,
            "usable": False,
            "rejection_stage": "answerability_required_ids",
            "reason": "Verifier did not return valid required_chunk_ids",
            "raw_generated": generated,
            "answerability_judgment": answerability,
        }

    generated_candidate_ids = [
        normalize_chunk_id(cid)
        for cid in generated.get("candidate_supporting_chunk_ids", [])
    ]

    generated_candidate_ids = [
        cid for cid in generated_candidate_ids
        if cid not in set(verified_required)
    ]

    acceptable_extra_ids, partial_support_ids, hard_negative_ids, support_judgments = verify_candidate_supports(
        page=page,
        generated=generated,
        candidate_ids=generated_candidate_ids,
    )
    
    acceptable_chunk_ids = sorted(
        set(verified_required) | set(acceptable_extra_ids),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    
    partial_support_chunk_ids = sorted(
        set(partial_support_ids),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    
    required_chunks = [
        {
            "chunk_id": cid,
            "chunk_index": chunk_map[cid].get("chunk_index"),
            "text": chunk_map[cid]["text"],
        }
        for cid in verified_required
    ]

    acceptable_chunks = [
        {
            "chunk_id": cid,
            "chunk_index": chunk_map[cid].get("chunk_index"),
            "text": chunk_map[cid]["text"],
        }
        for cid in acceptable_chunk_ids
    ]

    partial_support_chunks = [
        {
            "chunk_id": cid,
            "chunk_index": chunk_map[cid].get("chunk_index"),
            "text": chunk_map[cid]["text"],
        }
        for cid in partial_support_chunk_ids
    ]

    hard_negative_chunks = [
        {
            "chunk_id": cid,
            "chunk_index": chunk_map[cid].get("chunk_index"),
            "text": chunk_map[cid]["text"],
        }
        for cid in hard_negative_ids
    ]

    return {
        "page_id": page["page_id"],
        "url": page["url"],
        "usable": True,

        "query": query,
        "expected_answer": expected_answer,
        "question_type": generated.get("question_type"),
        "difficulty": generated.get("difficulty"),

        "source_window": {
            "window_id": source_window.get("window_id"),
            "chunk_ids": source_window.get("chunk_ids"),
            "chunk_indices": source_window.get("chunk_indices"),
            "window_size": source_window.get("window_size"),
            "start_chunk_index": source_window.get("start_chunk_index"),
            "end_chunk_index_exclusive": source_window.get("end_chunk_index_exclusive"),
            "position_bucket": source_window.get("position_bucket"),
        },

        "generation_source_chunk_ids": source_chunk_ids,
        "required_chunk_ids": verified_required,
        "acceptable_chunk_ids": acceptable_chunk_ids,
        "candidate_supporting_chunk_ids": generated_candidate_ids,
        "hard_negative_chunk_ids": hard_negative_ids,
        "partial_support_chunk_ids": partial_support_chunk_ids,

        "required_chunks": required_chunks,
        "acceptable_chunks": acceptable_chunks,
        "partial_support_chunks": partial_support_chunks,
        "hard_negative_chunks": hard_negative_chunks,

        "key_facts": generated.get("key_facts", []),

        "answerability_judgment": answerability,
        "support_judgments": support_judgments,
        "support_label_status": "llm_verified_pooled_not_exhaustive",
    }


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    if MAX_PAGES is not None:
        pages = pages[:MAX_PAGES]

    output = []
    rejected = []

    total_windows = sum(len(page.get("source_windows", [])) for page in pages)
    processed_windows = 0

    for page_i, page in enumerate(pages, start=1):
        source_windows = page.get("source_windows", [])

        if MAX_WINDOWS_PER_PAGE is not None:
            source_windows = source_windows[:MAX_WINDOWS_PER_PAGE]

        print(f"Processing page {page_i}/{len(pages)}: {page['url']}")
        print(f"  Source windows: {len(source_windows)}")

        for window_i, source_window in enumerate(source_windows, start=1):
            processed_windows += 1
            print(
                f"  Window {window_i}/{len(source_windows)} "
                f"({processed_windows}/{total_windows}): "
                f"{source_window.get('window_id')}"
            )

            try:
                result = generate_query_for_source_window(page, source_window)
            except Exception as e:
                print(f"  Error on page {page['page_id']}, window {source_window.get('window_id')}: {e}")
                continue

            if not result:
                continue

            if result.get("usable") is True:
                output.append(result)
            else:
                rejected.append(result)

    final = {
        "metadata": {
            "model": OPEN_AI_MODEL,
            "temperature": TEMPERATURE,
            "input_file": INPUT_FILE,
            "num_pages": len(pages),
            "num_usable_examples": len(output),
            "num_rejected_examples": len(rejected),
        },
        "examples": output,
        "rejected": rejected,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(
        f"Done. Saved {len(output)} usable examples and "
        f"{len(rejected)} rejected examples to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()