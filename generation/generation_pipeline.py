"""
generation/pipeline.py — End-to-end: query → retrieve → generate → store.
"""

import threading

from retrieval.retriever import embed_query, similarity_search, mmr_search, hybrid_search
from generation.prompt_builder import build_prompt, extract_sources, format_answer_with_sources
from generation.llm_handler import generate_answer, get_available_llms
from generation.answer_store import save_qa, save_comparison


def _save_qa_background(query, retrieval_output, generation_result, prompt, sources, session_id=None, user_id=None):
    """Save Q&A to PostgreSQL in a background thread (non-blocking)."""
    try:
        record_id = save_qa(query, retrieval_output, generation_result, prompt, sources, session_id, user_id)
        print(f"💾 Q&A saved to PostgreSQL (id: {record_id})")
    except Exception as e:
        print(f"⚠️  Background DB save failed: {e}")


def run_rag_query(query, qdrant_client, collection_name, embedding_model,
                  chunks, retriever_type="similarity", llm_key="phi3-mini",
                  top_k=5, session_id=None, user_id=None) -> dict:
    """Full RAG pipeline: retrieve → build prompt → generate → store.
    DB save runs in background thread so the response returns immediately."""
    query_vector = embed_query(query, embedding_model)

    retriever_map = {
        "similarity": lambda: similarity_search(qdrant_client, collection_name, query_vector, query, top_k),
        "mmr": lambda: mmr_search(qdrant_client, collection_name, query_vector, query, top_k),
        "hybrid": lambda: hybrid_search(qdrant_client, collection_name, query_vector, query, chunks, top_k),
    }

    if retriever_type not in retriever_map:
        retriever_type = "similarity"

    retrieval_output = retriever_map[retriever_type]()
    retrieved_chunks = retrieval_output.get("results", [])

    if not retrieved_chunks:
        return {
            "query": query, "retriever_type": retriever_type, "llm_key": llm_key, "top_k": top_k,
            "retrieved_chunks": [], "sources": [],
            "answer": "No relevant documents found.",
            "latency_retrieval": retrieval_output.get("latency_seconds", 0),
            "latency_generation": 0,
            "latency_total": retrieval_output.get("latency_seconds", 0),
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "record_id": None,
        }

    chat_history_str = ""
    if session_id and user_id:
        from core.database import get_session_history
        history = get_session_history(session_id, user_id)[-3:] # last 3 turns
        if history:
            turns = []
            for h in history:
                turns.append(f"User: {h['query_text']}\nDeepRAG: {h['answer_text']}")
            chat_history_str = "\n\n".join(turns)

    prompt = build_prompt(query, retrieved_chunks, chat_history_str)
    sources = extract_sources(retrieved_chunks)
    generation_result = generate_answer(prompt, llm_key)
    generation_result["answer"] = format_answer_with_sources(generation_result["answer"], sources)

    # Save to DB in background thread — don't block the response
    thread = threading.Thread(
        target=_save_qa_background,
        args=(query, retrieval_output, generation_result, prompt, sources, session_id, user_id),
        daemon=True,
    )
    thread.start()

    return {
        "query": query,
        "retriever_type": retriever_type,
        "llm_key": llm_key,
        "top_k": top_k,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
        "answer": generation_result["answer"],
        "latency_retrieval": retrieval_output.get("latency_seconds", 0),
        "latency_generation": generation_result.get("latency_seconds", 0),
        "latency_total": (retrieval_output.get("latency_seconds", 0) + generation_result.get("latency_seconds", 0)),
        "input_tokens": generation_result.get("input_tokens", 0),
        "output_tokens": generation_result.get("output_tokens", 0),
        "cost_usd": generation_result.get("cost_usd", 0),
        "record_id": 0,  # DB save is async, record_id not available immediately
    }


def stream_rag_query(query, qdrant_client, collection_name, embedding_model,
                     chunks, retriever_type="similarity", llm_key="groq-llama8b",
                     top_k=5, session_id=None, user_id=None):
    """Streaming RAG pipeline: retrieve → build prompt → stream tokens → save.
    Yields JSON-serializable dicts with 'event' and 'data' keys for SSE."""
    import json
    from generation.llm_handler import stream_answer

    # Phase 1: Retrieval (fast, do synchronously)
    query_vector = embed_query(query, embedding_model)

    retriever_map = {
        "similarity": lambda: similarity_search(qdrant_client, collection_name, query_vector, query, top_k),
        "mmr": lambda: mmr_search(qdrant_client, collection_name, query_vector, query, top_k),
        "hybrid": lambda: hybrid_search(qdrant_client, collection_name, query_vector, query, chunks, top_k),
    }

    if retriever_type not in retriever_map:
        retriever_type = "similarity"

    retrieval_output = retriever_map[retriever_type]()
    retrieved_chunks = retrieval_output.get("results", [])

    if not retrieved_chunks:
        yield {"event": "retrieval_done", "data": {"sources": [], "llm_name": llm_key, "retriever_type": retriever_type}}
        yield {"event": "token", "data": {"token": "No relevant documents found."}}
        yield {"event": "done", "data": {"latency_seconds": retrieval_output.get("latency_seconds", 0), "cost_usd": 0, "record_id": 0}}
        return

    chat_history_str = ""
    if session_id and user_id:
        from core.database import get_session_history
        history = get_session_history(session_id, user_id)[-3:] # last 3 turns
        if history:
            turns = []
            for h in history:
                turns.append(f"User: {h['query_text']}\nDeepRAG: {h['answer_text']}")
            chat_history_str = "\n\n".join(turns)

    prompt = build_prompt(query, retrieved_chunks, chat_history_str)
    sources = extract_sources(retrieved_chunks)

    # Send retrieval metadata to frontend immediately
    yield {
        "event": "retrieval_done",
        "data": {
            "sources": sources,
            "llm_name": llm_key,
            "retriever_type": retriever_type,
            "retrieval_latency": retrieval_output.get("latency_seconds", 0),
        }
    }

    # Phase 2: Stream LLM tokens
    full_answer = []
    generation_meta = {}

    for chunk in stream_answer(prompt, llm_key):
        if "token" in chunk:
            full_answer.append(chunk["token"])
            yield {"event": "token", "data": {"token": chunk["token"]}}
        if chunk.get("done"):
            generation_meta = chunk

    # Append source citations
    answer_text = "".join(full_answer)
    answer_with_sources = format_answer_with_sources(answer_text, sources)

    # Build generation_result for DB save
    generation_result = {
        "llm_key": generation_meta.get("llm_key", llm_key),
        "llm_name": generation_meta.get("llm_name", llm_key),
        "model_id": generation_meta.get("model_id", ""),
        "provider": generation_meta.get("provider", ""),
        "answer": answer_with_sources,
        "input_tokens": generation_meta.get("input_tokens", 0),
        "output_tokens": generation_meta.get("output_tokens", 0),
        "latency_seconds": generation_meta.get("latency_seconds", 0),
        "cost_usd": generation_meta.get("cost_usd", 0),
    }

    # Save to DB in background thread
    thread = threading.Thread(
        target=_save_qa_background,
        args=(query, retrieval_output, generation_result, prompt, sources, session_id, user_id),
        daemon=True,
    )
    thread.start()

    # Send the source-appended suffix and final metadata
    source_suffix = answer_with_sources[len(answer_text):]
    if source_suffix:
        yield {"event": "token", "data": {"token": source_suffix}}

    yield {
        "event": "done",
        "data": {
            "sources": sources,
            "llm_name": generation_meta.get("llm_name", llm_key),
            "latency_seconds": (
                retrieval_output.get("latency_seconds", 0) +
                generation_meta.get("latency_seconds", 0)
            ),
            "cost_usd": generation_meta.get("cost_usd", 0),
            "record_id": 0,
        }
    }


def run_multimodel_comparison(query, qdrant_client, collection_name,
                              embedding_model, chunks,
                              retriever_type="similarity", top_k=5,
                              llm_keys=None) -> dict:
    """Run same query through multiple LLMs. Retrieves chunks ONCE for a fair comparison."""
    if llm_keys is None:
        available = get_available_llms()
        llm_keys = [m["key"] for m in available]

    if not llm_keys:
        print("❌ No LLMs available for comparison.")
        return {}

    query_vector = embed_query(query, embedding_model)
    retriever_map = {
        "similarity": lambda: similarity_search(qdrant_client, collection_name, query_vector, query, top_k),
        "mmr": lambda: mmr_search(qdrant_client, collection_name, query_vector, query, top_k),
        "hybrid": lambda: hybrid_search(qdrant_client, collection_name, query_vector, query, chunks, top_k),
    }
    retrieval_output = retriever_map.get(retriever_type, retriever_map["similarity"])()
    retrieved_chunks = retrieval_output.get("results", [])

    if not retrieved_chunks:
        print("❌ No chunks retrieved. Cannot compare.")
        return {}

    prompt = build_prompt(query, retrieved_chunks)
    sources = extract_sources(retrieved_chunks)

    results = {}
    for llm_key in llm_keys:
        try:
            gen_result = generate_answer(prompt, llm_key)
            gen_result["answer"] = format_answer_with_sources(gen_result["answer"], sources)
            results[llm_key] = gen_result
            save_qa(query, retrieval_output, gen_result, prompt, sources)
        except Exception as e:
            print(f"  ⚠️  {llm_key} failed: {e}")
            results[llm_key] = {"llm_name": llm_key, "answer": f"Error: {e}", "latency_seconds": 0, "cost_usd": 0}

    save_comparison(query, retriever_type, top_k, results)

    print(f"\n{'Model':<22} | {'Latency':>8} | {'Cost':>6} | Answer Preview")
    print(f"{'-'*22}-+-{'-'*8}-+-{'-'*6}-+-{'-'*30}")
    for llm_key, result in results.items():
        name = result.get("llm_name", llm_key)[:21]
        lat = result.get("latency_seconds", 0)
        cost = result.get("cost_usd", 0)
        cost_str = "FREE" if cost == 0 else f"${cost:.4f}"
        preview = result.get("answer", "")[:30].replace("\n", " ")
        print(f"  {name:<20} | {lat:>6.2f}s | {cost_str:>6} | {preview}...")

    return results


def print_rag_result(result: dict):
    """Print formatted RAG result with sources and timing."""
    llm_key = result.get("llm_key", "")
    cost = result.get("cost_usd", 0)
    cost_str = "FREE" if cost == 0 else f"${cost:.6f}"

    print()
    print("═" * 56)
    print("  DeepRAG Answer")
    print("═" * 56)
    print(f"  Query     : {result.get('query', '')}")
    print(f"  LLM       : {llm_key} ({cost_str})")
    print(f"  Retriever : {result.get('retriever_type', '')} (k={result.get('top_k', 5)})")
    print(f"  Record ID : {result.get('record_id', 'N/A')}")
    print("─" * 56)

    print("  Retrieved Chunks:")
    for chunk in result.get("retrieved_chunks", [])[:5]:
        score = chunk.get("score", 0)
        print(f"  [{chunk.get('rank', '?')}] {chunk.get('filename', '')} — Page {chunk.get('page_number', 0)}  (score: {score:.3f})")

    print("─" * 56)
    print("  Answer:")
    for line in result.get("answer", "").split("\n"):
        print(f"  {line}")

    print("─" * 56)
    ret_lat = result.get("latency_retrieval", 0)
    gen_lat = result.get("latency_generation", 0)
    tot_lat = result.get("latency_total", 0)
    print(f"  ⏱  Retrieval: {ret_lat:.2f}s | Generation: {gen_lat:.2f}s | Total: {tot_lat:.2f}s")
    print(f"  🪙  Cost: {cost_str}")
    print("═" * 56)
    print()
