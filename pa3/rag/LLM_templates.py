import dspy
from rag_context import retrieve_context_loaded_model
import json
import re


class RAG_OneShot_Signature(dspy.Signature):
    """
    Si natančen asistent za odgovarjanje na vprašanja na podlagi podanega konteksta.

    Pravila:
    - Odgovori izključno na podlagi podanega konteksta.
    - Ne uporabljaj zunanjega znanja.
    - Če kontekst ne vsebuje dovolj informacij za odgovor, odgovori:
      "Na podlagi podanega konteksta tega ni mogoče ugotoviti."
    - Odgovori v slovenščini.
    - Odgovor naj bo kratek, dejstven in neposreden.
    - Vsako pomembno dejstveno trditev v odgovoru podpri s citatom v obliki [ChunkID: <id>].
    - Citiraj samo ChunkID odsekov, ki neposredno podpirajo navedeno trditev.
    - Ne navaj virov, ki so samo tematsko povezani, vendar ne podpirajo trditve.
    - Če se odseki med seboj razlikujejo ali si nasprotujejo, to jasno povej in citiraj ustrezne ChunkID-je.
    - Ne izmišljaj številk, datumov, imen, vzrokov ali posledic.

    Obvezna pravila za references_json:
    - Vedno izpolni polje references_json.
    - references_json mora biti veljaven JSON seznam kot niz.
    - Če v odgovoru uporabiš citat [ChunkID: 123], mora references_json vsebovati objekt za ta isti chunk_id.
    - Če references_json vsebuje chunk_id, mora biti isti ChunkID citiran tudi v odgovoru.
    - Za vsak citirani ChunkID prepiši URL iz istega odseka v kontekstu.
    - Ne dodajaj virov, ki niso citirani v odgovoru.
    - Ne dodajaj virov, ki niso prisotni v kontekstu.
    - Če odgovora ni mogoče podati na podlagi konteksta, mora biti references_json točno: []

    Oblika references_json:
    [
      {
        "chunk_id": "123",
        "url": "https://example.com/clanek"
      }
    ]

    Primer pravilne oblike odgovora:

    answer:
    "Napovedano je, da bo občina obnovila most do konca septembra [ChunkID: 101]."

    references_json:
    "[{\"chunk_id\": \"101\", \"url\": \"https://example.com/novice/obnova-mostu\"}]"

    Primer, kadar odgovor ni mogoč:
    answer:
    "Na podlagi podanega konteksta tega ni mogoče ugotoviti."

    references_json:
    "[]"
    """
    context: str = dspy.InputField(
        desc=(
            "Pridobljeni besedilni odseki. Vsak odsek vsebuje ChunkID, URL in besedilo. "
            "URL za references_json mora biti prepisan iz istega odseka kot citirani ChunkID."
        )
    )
    question: str = dspy.InputField(
        desc="Uporabnikovo vprašanje v slovenščini."
    )
    answer: str = dspy.OutputField(
        desc=(
            "Kratek, dejstven odgovor v slovenščini, izpeljan izključno iz konteksta. "
            "Vsaka pomembna dejstvena trditev mora imeti citat v obliki [ChunkID: <id>]."
        )
    )
    references_json: str = dspy.OutputField(
        desc=(
            "Vrni IZKLJUČNO surov veljaven JSON seznam kot niz. "
            "Ne dodaj oznake 'references_json:', ne dodaj narekovajev okoli celotnega izpisa, "
            "ne dodaj Markdowna, kode, pojasnil ali dodatnega besedila. "
            "Prvi znak mora biti '[' in zadnji znak mora biti ']'. "
            "Če answer vsebuje citat [ChunkID: 123], mora seznam vsebovati natanko objekt "
            "{\"chunk_id\": \"123\", \"url\": \"URL_IZ_ISTEGA_CHUNKA\"}. "
            "Če answer nima citatov ali odgovor ni mogoč, vrni natanko: []"
        )
    )
    
    

class RAG_ZeroShot_Signature(dspy.Signature):
    """
    Si natančen asistent za odgovarjanje na vprašanja na podlagi podanega konteksta.

    Pravila:
    - Odgovori izključno na podlagi podanega konteksta.
    - Ne uporabljaj zunanjega znanja.
    - Če kontekst ne vsebuje dovolj informacij za odgovor, odgovori:
      "Na podlagi podanega konteksta tega ni mogoče ugotoviti."
    - Odgovori v slovenščini.
    - Odgovor naj bo kratek, dejstven in neposreden.
    - Vsako pomembno dejstveno trditev v odgovoru podpri s citatom v obliki [ChunkID: <id>].
    - Citiraj samo ChunkID odsekov, ki neposredno podpirajo navedeno trditev.
    - Ne navaj virov, ki so samo tematsko povezani, vendar ne podpirajo trditve.
    - Če se odseki med seboj razlikujejo ali si nasprotujejo, to jasno povej in citiraj ustrezne ChunkID-je.
    - Ne izmišljaj številk, datumov, imen, vzrokov ali posledic.

    Obvezna pravila za references_json:
    - Vedno izpolni polje references_json.
    - references_json mora biti veljaven JSON seznam kot niz.
    - Če v odgovoru uporabiš citat [ChunkID: 123], mora references_json vsebovati objekt za ta isti chunk_id.
    - Če references_json vsebuje chunk_id, mora biti isti ChunkID citiran tudi v odgovoru.
    - Za vsak citirani ChunkID prepiši URL iz istega odseka v kontekstu.
    - Ne dodajaj virov, ki niso citirani v odgovoru.
    - Ne dodajaj virov, ki niso prisotni v kontekstu.
    - Če odgovora ni mogoče podati na podlagi konteksta, mora biti references_json točno: []
    - references_json mora biti formata: [{\"chunk_id\": \"<id>\", \"url\": \"<url>\"}].
    
    Primer oblike references_json:
    [
      {
        "chunk_id": "123",
        "url": "https://example.com/clanek"
      }
    ]
    
    """
    context: str = dspy.InputField(
        desc=(
            "Pridobljeni besedilni odseki. Vsak odsek vsebuje ChunkID, URL in besedilo. "
            "URL za references_json mora biti prepisan iz istega odseka kot citirani ChunkID."
        )
    )
    question: str = dspy.InputField(
        desc="Uporabnikovo vprašanje v slovenščini."
    )
    answer: str = dspy.OutputField(
        desc=(
            "Kratek, dejstven odgovor v slovenščini, izpeljan izključno iz konteksta. "
            "Vsaka pomembna dejstvena trditev mora imeti citat v obliki [ChunkID: <id>]."
        )
    )
    references_json: str = dspy.OutputField(
        desc=(
            "OBVEZNO: veljaven JSON seznam kot niz. "
            "Vsebovati mora natanko vse ChunkID-je, citirane v answer, skupaj z njihovimi URL-ji iz konteksta. "
            "Če answer nima citatov, mora biti: []"
        )
    )
    
    
    

def extract_cited_chunk_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[ChunkID:\s*([^\]]+)\]", answer or ""))

def run_signature_rag(
    question,
    signature,
    embedding_model,
    embedding_dim,
    reranking_model,
    num_candidates: int = 50,
    num_final: int = 3,
) -> dict | None:
    try:
        RAG_context = retrieve_context_loaded_model(
            question,
            loaded_model=embedding_model,
            loaded_dim=embedding_dim,
            loaded_reranker=reranking_model,
            num_candidates=num_candidates,
            num_final=num_final,
        )

        response = dspy.Predict(signature)(
            context=RAG_context,
            question=question,
        )

        answer = response.answer
        references_raw = getattr(response, "references_json", "[]")

        try:
            references = json.loads(references_raw)
        except json.JSONDecodeError:
            references = []

        cited_chunk_ids = extract_cited_chunk_ids(answer)
        referenced_chunk_ids = {
            str(ref.get("chunk_id"))
            for ref in references
            if isinstance(ref, dict) and ref.get("chunk_id") is not None
        }

        return {
            "question": question,
            "answer": answer,
            "references": references,
            "references_raw": references_raw,
            "cited_chunk_ids": sorted(cited_chunk_ids),
            "referenced_chunk_ids": sorted(referenced_chunk_ids),
            "citation_reference_match": cited_chunk_ids == referenced_chunk_ids,
            "context": RAG_context,
        }

    except Exception as e:
        print(f"Error: {e}")
        return None



class DirectQuerySignature(dspy.Signature):
    """
    Si natančen asistent za odgovarjanje na vprašanja.

    Pravila:
    - Odgovori v slovenščini.
    - Odgovor naj bo kratek, dejstven in neposreden.
    - Vedno poskusi odgovoriti na podlagi svojega znanja.
    - Če odgovor temelji na splošnem znanju in ni popolnoma zanesljiv, ga jasno označi z: "Na podlagi splošnega znanja:"
    - Samo če vprašanje zahteva zelo specifičen dokument, člen ali svež podatek, ki ga absolutno ne moreš vedeti, odgovori:
      "Brez dodatnega konteksta tega ni mogoče zanesljivo ugotoviti."
    - Ne navajaj virov, citatov, ChunkID-jev ali URL-jev.
    """
    question: str = dspy.InputField(
        desc="Uporabnikovo vprašanje v slovenščini."
    )
    answer: str = dspy.OutputField(
        desc="Odgovor v slovenščini. Vedno poskusi odgovoriti. Če je odgovor splošen, ga označi z 'Na podlagi splošnega znanja:'. Le v skrajnem primeru uporabi dogovorjeni stavek o pomanjkanju konteksta."
    )

def run_signature_direct_query(question: str) -> str:
    try:
        response = dspy.Predict(DirectQuerySignature)(question=question)
        return response.answer
    except Exception as e:
        print(f"Error: {e}")
        return None