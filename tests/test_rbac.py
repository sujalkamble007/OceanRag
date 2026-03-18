"""
test_rbac.py — Tests all 4 roles against their allowed LLMs + retrievers.
Registers a user per role, logs in, and fires queries.
"""
import requests
import uuid
import json

BASE = "http://localhost:8000"

ROLES = {
    "common_user": {
        "allowed_llms": ["groq-llama8b"],
        "blocked_llms": ["groq-llama70b", "zephyr-7b"],
        "allowed_retrievers": ["similarity"],
        "blocked_retrievers": ["mmr", "hybrid"],
        "max_top_k": 3,
    },
    "student": {
        "allowed_llms": ["groq-llama8b", "groq-llama70b", "zephyr-7b"],
        "blocked_llms": ["qwen-72b"],
        "allowed_retrievers": ["similarity", "mmr"],
        "blocked_retrievers": ["hybrid"],
        "max_top_k": 5,
    },
    "researcher": {
        "allowed_llms": ["groq-llama8b", "groq-llama70b"],
        "blocked_llms": [],
        "allowed_retrievers": ["similarity", "mmr", "hybrid"],
        "blocked_retrievers": [],
        "max_top_k": 10,
    },
    "admin": {
        "allowed_llms": ["groq-llama8b"],
        "blocked_llms": [],
        "allowed_retrievers": ["similarity", "mmr"],
        "blocked_retrievers": [],
        "max_top_k": 10,
    },
}

QUERY = "What is UNCLOS?"
results = []


def register_and_login(role):
    uid = uuid.uuid4().hex[:6]
    email = f"test_{role}_{uid}@test.com"
    pwd = "TestPass123"
    uname = f"test_{role}_{uid}"

    # Register
    r = requests.post(f"{BASE}/auth/register", json={
        "email": email, "username": uname, "password": pwd, "role": role
    })
    if r.status_code not in (200, 201):
        return None, f"Register failed ({r.status_code}): {r.text}"

    # Login
    r = requests.post(f"{BASE}/auth/login", json={"email": email, "password": pwd})
    if r.status_code != 200:
        return None, f"Login failed ({r.status_code}): {r.text}"

    return r.json()["access_token"], None


def test_query(token, role, llm_key, retriever, top_k, expect_success):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "query": QUERY,
        "llm_key": llm_key,
        "retriever_type": retriever,
        "top_k": top_k,
    }
    try:
        r = requests.post(f"{BASE}/query", json=payload, headers=headers, timeout=30)
    except Exception as e:
        return {"role": role, "llm": llm_key, "retriever": retriever, "top_k": top_k,
                "expected": "PASS" if expect_success else "BLOCK",
                "actual": "ERROR", "detail": str(e)}

    if expect_success:
        passed = r.status_code == 200
        status = "✅ PASS" if passed else f"❌ FAIL ({r.status_code})"
    else:
        passed = r.status_code == 403
        status = "✅ BLOCKED" if passed else f"❌ SHOULD BLOCK ({r.status_code})"

    detail = ""
    if not passed:
        try:
            detail = r.json().get("detail", "")[:100]
        except:
            detail = r.text[:100]

    return {"role": role, "llm": llm_key, "retriever": retriever, "top_k": top_k,
            "expected": "PASS" if expect_success else "BLOCK",
            "actual": status, "detail": detail}


def main():
    print("=" * 70)
    print("  RBAC TEST — All Roles × LLMs × Retrievers")
    print("=" * 70)

    all_results = []

    for role, config in ROLES.items():
        print(f"\n{'─' * 60}")
        print(f"  ROLE: {role.upper()}")
        print(f"{'─' * 60}")

        token, err = register_and_login(role)
        if err:
            print(f"  ⚠️  {err}")
            continue

        print(f"  ✅ Registered & logged in")

        # Test allowed LLMs with first allowed retriever
        first_ret = config["allowed_retrievers"][0]
        for llm in config["allowed_llms"]:
            print(f"  Testing ALLOWED: llm={llm}, ret={first_ret} ...", end=" ", flush=True)
            res = test_query(token, role, llm, first_ret, config["max_top_k"], True)
            print(res["actual"])
            all_results.append(res)

        # Test blocked LLMs
        for llm in config["blocked_llms"]:
            print(f"  Testing BLOCKED LLM: llm={llm}, ret={first_ret} ...", end=" ", flush=True)
            res = test_query(token, role, llm, first_ret, config["max_top_k"], False)
            print(res["actual"])
            all_results.append(res)

        # Test allowed retrievers with first allowed LLM
        first_llm = config["allowed_llms"][0]
        for ret in config["allowed_retrievers"]:
            print(f"  Testing ALLOWED: llm={first_llm}, ret={ret} ...", end=" ", flush=True)
            res = test_query(token, role, first_llm, ret, config["max_top_k"], True)
            print(res["actual"])
            all_results.append(res)

        # Test blocked retrievers
        for ret in config["blocked_retrievers"]:
            print(f"  Testing BLOCKED retriever: llm={first_llm}, ret={ret} ...", end=" ", flush=True)
            res = test_query(token, role, first_llm, ret, config["max_top_k"], False)
            print(res["actual"])
            all_results.append(res)

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for r in all_results if "✅" in r["actual"])
    failed = sum(1 for r in all_results if "❌" in r["actual"])
    total = len(all_results)

    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")

    if failed > 0:
        print(f"\n  ❌ FAILURES:")
        for r in all_results:
            if "❌" in r["actual"]:
                print(f"    Role={r['role']}, LLM={r['llm']}, Ret={r['retriever']}: {r['actual']} — {r['detail']}")
    else:
        print(f"\n  🎉 All {total} tests passed!")


if __name__ == "__main__":
    main()
