from __future__ import annotations

from typing import Dict

# Placeholders: {context}, {question}, {history}
PROMPT_TEMPLATES: Dict[str, str] = {
    "default": """Use the following passages to answer the question. Be concise; answer with a short phrase or sentence when possible.

History:
{history}

Passages:
{context}

Question: {question}
Answer:""",
    "bullets": """Read the context and answer the question in a few words.

History:
{history}

Context:
{context}

Q: {question}
A:""",
    "strict_cite": """Answer only using information from the context below. If the answer is not in the context, say "unknown".
Every factual claim must include at least one citation in [n] format where n is the passage number.
If multiple passages support the answer, include multiple citations like [1][3].

History:
{history}

Context:
{context}

Question: {question}
Your answer (short, with citations):""",
}


def format_rag_prompt(
    template: str, *, context: str, question: str, history: str = "None"
) -> str:
    return template.format(context=context, question=question, history=history)
