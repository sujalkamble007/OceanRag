"""
llm_handler.py — Unified LLM interface for Phase 3.
HuggingFace Inference API (FREE) is primary. Groq/Anthropic are optional.
"""

import os
import time
import requests as http_requests

from prompt_builder import build_hf_prompt_string


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
    # ── FREE: HuggingFace Inference API ──────────────────────
    "qwen2.5-72b": {
        "name": "Qwen 2.5 72B Instruct",
        "provider": "huggingface",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "hf_url": "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
    "mistral-nemo": {
        "name": "Mistral Nemo 12B",
        "provider": "huggingface",
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "hf_url": "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
    "llama3.1-8b": {
        "name": "LLaMA 3.1 8B (HF)",
        "provider": "huggingface",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "hf_url": "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
    "gemma2-2b": {
        "name": "Gemma 2 2B (HF)",
        "provider": "huggingface",
        "model_id": "google/gemma-2-2b-it",
        "hf_url": "https://api-inference.huggingface.co/models/google/gemma-2-2b-it",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
    "phi3.5-mini": {
        "name": "Phi 3.5 Mini (HF)",
        "provider": "huggingface",
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "hf_url": "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct",
        "input_cost_per_1k": 0.0,
        "output_cost_per_1k": 0.0,
        "max_tokens": 500,
        "requires_key": "HF_API_TOKEN",
    },
}


# ─── Function 1: Get Available LLMs ─────────────────────────────────────────

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


# ─── Function 2: Call HuggingFace ────────────────────────────────────────────

def call_huggingface(prompt: dict, llm_key: str, max_tokens: int) -> dict:
    """
    Call HuggingFace Inference API with retry for cold starts and rate limits.
    """
    hf_token = os.getenv("HF_API_TOKEN", "").strip()
    if not hf_token:
        raise ValueError(
            "HF_API_TOKEN not set. Get free token from huggingface.co/settings/tokens"
        )

    config = LLM_CONFIGS[llm_key]
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Build single prompt string for HF API
    # Use the retrieved chunks from the prompt dict
    full_prompt = prompt["system"] + "\n\n" + prompt["user"]

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }

    try:
        resp = http_requests.post(config["hf_url"], headers=headers, json=payload, timeout=60)

        # Handle cold start (model loading)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                error_msg = str(data["error"]).lower()
                if "loading" in error_msg or "currently loading" in error_msg:
                    print("  ⏳ Model loading on HF servers (~20-30s first call)...")
                    time.sleep(25)
                    resp = http_requests.post(
                        config["hf_url"], headers=headers, json=payload, timeout=60
                    )
        elif resp.status_code == 503:
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if "loading" in str(body).lower():
                print("  ⏳ Model loading on HF servers (~20-30s first call)...")
                time.sleep(25)
                resp = http_requests.post(
                    config["hf_url"], headers=headers, json=payload, timeout=60
                )

        # Handle rate limit
        if resp.status_code == 429:
            print("  ⚠️  HuggingFace rate limit hit. Waiting 60s...")
            time.sleep(60)
            resp = http_requests.post(
                config["hf_url"], headers=headers, json=payload, timeout=60
            )

        resp.raise_for_status()

        # Parse response
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            result = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            result = data.get("generated_text", str(data))
        else:
            result = str(data)

        # Strip prompt echo if return_full_text was ignored
        if result.startswith(full_prompt):
            result = result[len(full_prompt):]

        result = result.strip()

        return {"answer": result, "input_tokens": 0, "output_tokens": 0}

    except Exception as e:
        return {"answer": f"HuggingFace API error: {str(e)}", "input_tokens": 0, "output_tokens": 0}


# ─── Function 3: Call Groq ───────────────────────────────────────────────────

def call_groq(prompt: dict, model_id: str, max_tokens: int) -> dict:
    """Call Groq API. Uses OpenAI SDK with Groq base_url."""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
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


# ─── Function 4: Calculate Cost ─────────────────────────────────────────────

def calculate_cost(llm_key: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD. Returns 0.0 for HuggingFace (always free)."""
    config = LLM_CONFIGS[llm_key]
    if config["input_cost_per_1k"] == 0 and config["output_cost_per_1k"] == 0:
        return 0.0
    return (input_tokens / 1000 * config["input_cost_per_1k"] +
            output_tokens / 1000 * config["output_cost_per_1k"])


# ─── Function 6: Generate Answer (Main Interface) ───────────────────────────

def generate_answer(prompt: dict, llm_key: str) -> dict:
    """
    Unified interface: route to the correct LLM provider.
    All other code calls ONLY this function.
    """
    if llm_key not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM key: {llm_key}. Available: {list(LLM_CONFIGS.keys())}")

    config = LLM_CONFIGS[llm_key]

    # Check availability
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


# ─── Function 7: Print LLM Response ─────────────────────────────────────────

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
