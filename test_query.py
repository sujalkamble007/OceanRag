import requests

login = requests.post("http://localhost:8000/auth/login", data={"username": "testuser", "password": "password123"})
token = login.json()["access_token"]

resp = requests.post("http://localhost:8000/query/", 
    headers={"Authorization": f"Bearer {token}"},
    json={"query": "Hi what is UNCLOS", "llm_key": "groq-llama8b", "retriever_type": "mmr", "top_k": 3}
)
print(resp.status_code)
print(resp.json())
