from __future__ import annotations

from typing import Dict

# Placeholders: {context}, {question}
PROMPT_TEMPLATES: Dict[str, str] = {
    "default": """Use the following passages to answer the question. Be concise; answer with a short phrase or sentence when possible.

Passages:
{context}

Question: {question}
Answer:""",
    "bullets": """Read the context and answer the question in a few words.

Context:
{context}

Q: {question}
A:""",
    "strict_cite": """Answer only using information from the context below. If the answer is not in the context, say "unknown".

Context:
{context}

Question: {question}
Your answer (short):""",
}


def format_rag_prompt(template: str, *, context: str, question: str) -> str:
    return template.format(context=context, question=question)
