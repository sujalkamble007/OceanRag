"""
generation/prompt_builder.py — Builds structured prompts from retrieved chunks.
Used by Phase 3 LLM Generation to create grounded, cited answers.
"""


# ─── Prompt Templates ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DeepRAG, an expert AI assistant specialized in deep-sea \
governance, international maritime law, and ocean regulation documents.

Your role is to answer questions STRICTLY based on the provided context documents.

RULES YOU MUST FOLLOW:
1. For queries related to deep-sea governance, mining, or ocean regulations, answer ONLY using information present in the provided context.
2. If the user's input is a casual greeting or small talk (e.g., "hi", "hello", "how are you"), respond naturally as a helpful AI assistant.
3. If the user asks a general knowledge question clearly unrelated to the domain documents, answer it helpfully, but add a brief note that this information is from your general knowledge and not the context documents.
4. For domain-specific factual queries, if the answer is not in the context, respond with exactly:
   "I could not find sufficient information in the provided documents to answer this question."
5. Always cite your sources at the end for factual claims using this format:
   [Source: <filename>, Page <page_number>]
6. If multiple sources support your answer, cite all of them.
7. Never make up information or use external knowledge for domain-specific questions.
8. Be precise, professional, and concise.
"""

CONTEXT_TEMPLATE = """--- Document Chunk {i} ---
Source: {filename} | Page: {page_number}
Content:
{page_content}
"""

USER_PROMPT_TEMPLATE = """Based ONLY on the following document excerpts, answer the question.

=== RETRIEVED CONTEXT ===
{context_block}
=== END CONTEXT ===

{chat_history_block}

Current Question: {query}

Answer (cite your sources):"""



def build_context_block(retrieved_chunks: list) -> str:
    """Format retrieved chunks into a single context string."""
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        block = CONTEXT_TEMPLATE.format(
            i=i,
            filename=chunk.get("filename", "unknown"),
            page_number=chunk.get("page_number", 0),
            page_content=chunk.get("page_content", ""),
        )
        blocks.append(block)

    context = "\n\n".join(blocks)
    total_chars = len(context)
    print(f"📄 Built context from {len(retrieved_chunks)} chunks ({total_chars} chars)")
    return context


def build_prompt(query: str, retrieved_chunks: list, chat_history: str = "") -> dict:
    """Build a structured prompt dict with system and user messages."""
    context_block = build_context_block(retrieved_chunks)
    history_block = f"=== CHAT HISTORY ===\n{chat_history}\n=== END HISTORY ===" if chat_history else ""
    user_prompt = USER_PROMPT_TEMPLATE.format(context_block=context_block, chat_history_block=history_block, query=query)

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "context_block": context_block,
        "chunk_count": len(retrieved_chunks),
        "total_chars": len(context_block),
    }


def extract_sources(retrieved_chunks: list) -> list:
    """Extract unique (filename, page_number) pairs from retrieved chunks."""
    seen = set()
    sources = []
    for chunk in retrieved_chunks:
        key = (chunk.get("filename", ""), chunk.get("page_number", 0))
        if key not in seen:
            seen.add(key)
            sources.append({"filename": key[0], "page_number": key[1]})
    return sources


def format_answer_with_sources(answer: str, sources: list) -> str:
    """Append source citations to the answer if not already present."""
    if "[Source:" in answer or "📚 Sources:" in answer:
        return answer

    source_lines = [f"  • {s['filename']} — Page {s['page_number']}" for s in sources]
    return answer.rstrip() + "\n\n📚 Sources:\n" + "\n".join(source_lines)


def build_hf_prompt_string(query: str, retrieved_chunks: list, chat_history: str = "") -> str:
    """Build a single string prompt for HuggingFace Inference API."""
    context_block = build_context_block(retrieved_chunks)
    history_block = f"=== CHAT HISTORY ===\n{chat_history}\n=== END HISTORY ===\n" if chat_history else ""
    return f"""{SYSTEM_PROMPT}

{context_block}

{history_block}
Current Question: {query}
Answer:"""
