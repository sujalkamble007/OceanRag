"""
generation/pipeline.py — End-to-end: query → retrieve → generate → store.
"""

from retrieval.retriever import embed_query, similarity_search, mmr_search, hybrid_search
from generation.prompt_builder import build_prompt, extract_sources, format_answer_with_sources
from generation.llm_handler import generate_answer, get_available_llms
from generation.answer_store import save_qa, save_comparison


def run_rag_query(query, qdrant_client, collection_name, embedding_model,
                  chunks, retriever_type="similarity", llm_key="phi3-mini",
                  top_k=5) -> dict:
    """Full RAG pipeline: retrieve → build prompt → generate → store."""
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

    prompt = build_prompt(query, retrieved_chunks)
    sources = extract_sources(retrieved_chunks)
    generation_result = generate_answer(prompt, llm_key)
    generation_result["answer"] = format_answer_with_sources(generation_result["answer"], sources)
    record_id = save_qa(query, retrieval_output, generation_result, prompt, sources)

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
        "record_id": record_id,
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
