import json
import urllib.request
import urllib.error

def test_api():
    print("=== API LAYER ===")
    
    # 1. GET /graph/stats
    try:
        req = urllib.request.Request("http://localhost:8000/graph/stats")
        res = urllib.request.urlopen(req)
        stats = json.loads(res.read())
        print("GET /graph/stats -> Pass:", stats["total_nodes"] == 1338)
    except Exception as e:
        print("GET /graph/stats -> Fail:", e)

    # 2. GET /graph/node/bad
    try:
        req = urllib.request.Request("http://localhost:8000/graph/node/Customer:bad")
        res = urllib.request.urlopen(req)
        print("GET /graph/node/{node_id} (non-existent) -> Fail: Expected 404, got 200")
    except urllib.error.HTTPError as e:
        print("GET /graph/node/{node_id} (non-existent) -> Pass:", e.code == 404)

    # 3. GET /graph/search
    try:
        req = urllib.request.Request("http://localhost:8000/graph/search?q=Cardenas")
        res = urllib.request.urlopen(req)
        data = json.loads(res.read())
        print("GET /graph/search?q=Cardenas -> Pass:", len(data["results"]) > 0 and '310000108' in data["results"][0]["id"])
    except Exception as e:
        print("GET /graph/search?q=Cardenas -> Fail:", e)
        
    # 4. POST /chat empty message
    try:
        body = json.dumps({"messages": []}).encode("utf-8")
        req = urllib.request.Request("http://localhost:8000/chat", data=body, headers={"Content-Type": "application/json"})
        res = urllib.request.urlopen(req)
        print("POST /chat empty message -> Fail: Expected 400")
    except urllib.error.HTTPError as e:
        print("POST /chat empty message -> Pass:", e.code == 400)


def ask_chat(msg):
    req = urllib.request.Request("http://localhost:8000/chat", data=json.dumps({"messages": [{"role": "user", "content": msg}]}).encode("utf-8"), headers={"Content-Type": "application/json"})
    out = ""
    nodes = []
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    js = line_str[6:]
                    if js.strip() == "[DONE]": break
                    try:
                        d = json.loads(js)
                        if "token" in d:
                            out += d["token"]
                        if "referenced_nodes" in d:
                            nodes = d["referenced_nodes"]
                    except: pass
    except Exception as e:
        return f"Error: {e}", []
    return out, nodes

def test_chat_prompts():
    print("\n=== CHAT QUERIES ===")
    
    queries = [
        "Which products are associated with the highest number of billing documents?",
        "Trace the full flow of billing document 90504248",
        "Identify sales orders that have been delivered but not billed",
        "Identify sales orders that are billed but have no delivery document",
        "Search for customer Cardenas",
        "Search for product TG",
        "Search for customer jkxqzw"
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        ans, nodes = ask_chat(q)
        print(f"A: {ans}")
        print(f"Referenced Nodes: {len(nodes)} max cap check")
        
    print("\n=== GUARDRAILS ===")
    guards = [
        "What is the capital of France?",
        "Write me a poem",
        "What is 2 + 2?"
    ]
    for q in guards:
        print(f"\nQ: {q}")
        ans, _ = ask_chat(q)
        print(f"A: {ans}")

test_api()
test_chat_prompts()
