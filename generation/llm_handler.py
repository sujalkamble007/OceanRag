"""
generation/llm_handler.py — Unified LLM interface for Phase 3.
HuggingFace Inference API (FREE) is primary. Groq is optional but faster.
"""

import os
import time
import requests as http_requests

from generation.prompt_builder import build_hf_prompt_string


# ─── LLM Configurations ─────────────────────────────────────────────────────

LLM_CONFIGS = {
    # ── Groq (fast, reliable) ────────────────────────────────
    "groq-llama70b": {
        "name": "LLaMA 3.3 70B (Groq)",
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "input_cost_per_1k": 0.00059,
        "output_cost_per_1k": 0.00079,
        "max_tokens": 500,
        "requires_key": "GROQ_API_KEY",
    },
    "groq-llama8b": {
        "name": "LLaMA 3.1 8B (Groq)",
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "input_cost_per_1k": 0.00005,
        "output_cost_per_1k": 0.00008,
        "max_tokens": 500,
        "requires_key": "GROQ_API_KEY",
    },
    "groq-qwen32b": {
        "name": "Qwen 3 32B (Groq)",
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
        "input_cost_per_1k": 0.0002,
        "output_cost_per_1k": 0.0002,
        "max_tokens": 500,
        "requires_key": "GROQ_API_KEY",
    },
    "groq-llama4-17b": {
        "name": "LLaMA 4 17B Scout (Groq)",
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "input_cost_per_1k": 0.0003,
        "output_cost_per_1k": 0.0003,
        "max_tokens": 500,
        "requires_key": "GROQ_API_KEY",
    },
    "groq-kimi": {
        "name": "Kimi K2 (Groq)",
        "provider": "groq",
        "model_id": "moonshotai/kimi-k2-instruct",
        "input_cost_per_1k": 0.00024,
        "output_cost_per_1k": 0.00024,
        "max_tokens": 500,
        "requires_key": "GROQ_API_KEY",
    },
    # ── FREE: HuggingFace Inference API (via InferenceClient SDK) ────
    "qwen2.5-72b": {
        "name": "Qwen 2.5 72B (HF)",
        "provider": "huggingface",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
    "zephyr-7b": {
        "name": "Zephyr 7B (HF)",
        "provider": "huggingface",
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
    "gemma-2-9b": {
        "name": "Gemma 2 9B (HF)",
        "provider": "huggingface",
        "model_id": "google/gemma-2-9b-it",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
    "llama3.2-3b": {
        "name": "LLaMA 3.2 3B (HF)",
        "provider": "huggingface",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
}


def get_available_llms() -> list:
    """Check which LLMs are available based on env var keys."""
    available = []
    print("\nAvailable LLMs:")
    for key, config in LLM_CONFIGS.items():
        env_key = config["requires_key"]
        token = os.getenv(env_key, "").strip()
        if token:
            provider_label = config["provider"].capitalize()
            cost_label = "FREE" if config["input_cost_per_1k"] == 0 else "PAID"
            print(f"  ✅ {key:<16} — {config['name']:<26} ({provider_label} - {cost_label})")
            available.append({**config, "key": key})
        else:
            print(f"  ❌ {key:<16} — {config['name']:<26} (No {env_key} in .env)")
    print()
    return available


def call_huggingface(prompt: dict, llm_key: str, max_tokens: int) -> dict:
    """Call HuggingFace via InferenceClient SDK (auto-resolves correct endpoint)."""
    hf_token = os.getenv("HF_API_TOKEN", "").strip()
    if not hf_token:
        raise ValueError("HF_API_TOKEN not set. Get free token from huggingface.co/settings/tokens")

    config = LLM_CONFIGS[llm_key]
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=hf_token)
        result = client.chat.completions.create(
            model=config["model_id"],
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user",   "content": prompt["user"]},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        answer = result.choices[0].message.content.strip()
        return {"answer": answer, "input_tokens": 0, "output_tokens": 0}
    except Exception as e:
        return {"answer": f"HuggingFace API error: {str(e)}", "input_tokens": 0, "output_tokens": 0}


def call_groq(prompt: dict, model_id: str, max_tokens: int) -> dict:
    """Call Groq API. Uses OpenAI SDK with Groq base_url."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return {
            "answer": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        return {"answer": f"Groq API error: {str(e)}", "input_tokens": 0, "output_tokens": 0}


def calculate_cost(llm_key: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD. Returns 0.0 for HuggingFace (always free)."""
    config = LLM_CONFIGS[llm_key]
    if config["input_cost_per_1k"] == 0 and config["output_cost_per_1k"] == 0:
        return 0.0
    return (input_tokens / 1000 * config["input_cost_per_1k"] +
            output_tokens / 1000 * config["output_cost_per_1k"])


def generate_answer(prompt: dict, llm_key: str) -> dict:
    """Unified interface: route to the correct LLM provider."""
    if llm_key not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM key: {llm_key}. Available: {list(LLM_CONFIGS.keys())}")

    config = LLM_CONFIGS[llm_key]
    env_key = config["requires_key"]
    if not os.getenv(env_key, "").strip():
        raise ValueError(f"LLM '{llm_key}' requires {env_key} in .env")

    max_tokens = config["max_tokens"]
    provider = config["provider"]
    print(f"🤖 Calling {config['name']} ({provider})...")

    start = time.time()
    if provider == "huggingface":
        result = call_huggingface(prompt, llm_key, max_tokens)
    elif provider == "groq":
        result = call_groq(prompt, config["model_id"], max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    end = time.time()
    cost = calculate_cost(llm_key, result["input_tokens"], result["output_tokens"])

    return {
        "llm_key": llm_key,
        "llm_name": config["name"],
        "model_id": config["model_id"],
        "provider": provider,
        "answer": result["answer"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_seconds": round(end - start, 3),
        "cost_usd": cost,
    }


def stream_groq(prompt: dict, model_id: str, max_tokens: int):
    """Stream tokens from Groq API. Yields (token_str, None) or (None, usage_dict)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        stream = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content, None
            # Final chunk carries usage stats
            if chunk.usage:
                yield None, {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
    except Exception as e:
        yield f"Groq API error: {str(e)}", None


def stream_huggingface(prompt: dict, llm_key: str, max_tokens: int):
    """Stream tokens from HuggingFace InferenceClient. Yields (token_str, None)."""
    hf_token = os.getenv("HF_API_TOKEN", "").strip()
    if not hf_token:
        yield "HF_API_TOKEN not set.", None
        return

    config = LLM_CONFIGS[llm_key]
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=hf_token)
        stream = client.chat.completions.create(
            model=config["model_id"],
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user",   "content": prompt["user"]},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content, None
    except Exception as e:
        yield f"HuggingFace API error: {str(e)}", None


def stream_answer(prompt: dict, llm_key: str):
    """Unified streaming interface. Yields dicts: {"token": str} or {"done": True, ...metadata}."""
    if llm_key not in LLM_CONFIGS:
        yield {"token": f"Unknown LLM key: {llm_key}"}
        yield {"done": True, "llm_key": llm_key, "llm_name": llm_key, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0, "cost_usd": 0}
        return

    config = LLM_CONFIGS[llm_key]
    env_key = config["requires_key"]
    if not os.getenv(env_key, "").strip():
        yield {"token": f"LLM '{llm_key}' requires {env_key} in .env"}
        yield {"done": True, "llm_key": llm_key, "llm_name": config["name"], "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0, "cost_usd": 0}
        return

    max_tokens = config["max_tokens"]
    provider = config["provider"]
    print(f"🤖 Streaming {config['name']} ({provider})...")

    start = time.time()
    usage = {"input_tokens": 0, "output_tokens": 0}

    if provider == "groq":
        for token, token_usage in stream_groq(prompt, config["model_id"], max_tokens):
            if token:
                yield {"token": token}
            if token_usage:
                usage = token_usage
    elif provider == "huggingface":
        for token, _ in stream_huggingface(prompt, llm_key, max_tokens):
            if token:
                yield {"token": token}
    else:
        yield {"token": f"Unknown provider: {provider}"}

    end = time.time()
    cost = calculate_cost(llm_key, usage["input_tokens"], usage["output_tokens"])

    yield {
        "done": True,
        "llm_key": llm_key,
        "llm_name": config["name"],
        "model_id": config["model_id"],
        "provider": provider,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "latency_seconds": round(end - start, 3),
        "cost_usd": cost,
    }


def print_llm_response(generation_result: dict):
    """Print a formatted LLM response box."""
    name = generation_result["llm_name"]
    latency = generation_result["latency_seconds"]
    cost = generation_result["cost_usd"]
    cost_str = "FREE" if cost == 0 else f"${cost:.6f}"

    print()
    print("┌" + "─" * 58 + "┐")
    print(f"│  LLM: {name:<20} │  Latency: {latency}s  │  Cost: {cost_str:<8} │")
    print("└" + "─" * 58 + "┘")
    print()
    print("Answer:")
    print(generation_result["answer"])
    print()

    tokens = generation_result["input_tokens"] + generation_result["output_tokens"]
    if tokens > 0:
        print(f"Tokens: {generation_result['input_tokens']} in / {generation_result['output_tokens']} out")
    else:
        print("Tokens: N/A (HuggingFace free tier)")
