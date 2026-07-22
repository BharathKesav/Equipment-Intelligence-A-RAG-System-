# src/generation/generator.py — Groq version

import os
import re
import yaml
from groq import Groq


def load_prompts(config_path: str = "configs/prompts.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_context_string(top_chunks: list[dict]) -> tuple[str, list[dict]]:
    # Identical to before — no changes
    context_parts = []
    citation_map  = []
    for i, chunk in enumerate(top_chunks):
        number = i + 1
        passage = (
            f"[{number}] "
            f"(Source: {chunk['metadata'].get('paragraph_ref', 'unknown')}):\n"
            f"{chunk['text']}"
        )
        context_parts.append(passage)
        citation_map.append({
            "number":        number,
            "paragraph_ref": chunk["metadata"].get("paragraph_ref", "unknown"),
            "source":        chunk["metadata"].get("source", "unknown"),
            "page":          chunk["metadata"].get("page", 0) + 1,
            "chunk_index":   chunk["metadata"].get("chunk_index", -1),
        })
    return "\n\n".join(context_parts), citation_map


def generate_answer(
    query: str,
    top_chunks: list[dict],
    top_scores: list[float],
) -> dict:

    prompts = load_prompts()
    context_string, citation_map = build_context_string(top_chunks)

    user_message = prompts["user_prompt_template"].format(
        context=context_string,
        question=query,
    )

    # ── Groq API call ──
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model=prompts.get("llm_model", "llama-3.1-8b-instant"),
        temperature=0,
        messages=[
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user",   "content": user_message},
        ],
    )

    answer_text = response.choices[0].message.content

    used_numbers   = set(int(n) for n in re.findall(r'\[(\d+)\]', answer_text))
    used_citations = [c for c in citation_map if c["number"] in used_numbers]

    return {
        "answer":     answer_text,
        "citations":  used_citations,
        "declined":   False,
        "confidence": float(max(top_scores)) if top_scores else 0.0,
        "prompt_ver": prompts.get("version", "unknown"),
    }