from duckduckgo_search import DDGS
from ollama import chat

model = "qwen3:0.6b"

# basic wrapper to search ddg
def search_ddg(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return [r['body'] for r in results][:max_results]

# ask ollama if we need to search
def should_search(prompt):
    resp = chat(model=model, messages=[
        {"role": "system", "content": "you are an AI assistant that decides if a prompt needs an internet search. answer only YES or NO."},
        {"role": "user", "content": prompt}
    ])
    return "yes" in resp["message"]["content"].lower()

# summarize search results w/ ollama
def summarize_with_ollama(query, results):
    text = "\n".join(f"- {r}" for r in results)
    resp = chat(model=model, messages=[
        {"role": "system", "content": "you are an AI assistant that summarizes search results."},
        {"role": "user", "content": f"here are the search results about '{query}':\n{text}\n\nplease summarize them."}
    ])
    return resp["message"]["content"]

# full agent
def ask_agent(prompt):
    if should_search(prompt):
        results = search_ddg(prompt)
        summary = summarize_with_ollama(prompt, results)
        return summary
    else:
        resp = chat(model=model, messages=[
            {"role": "user", "content": prompt}
        ])
        return resp["message"]["content"]

# example
if __name__ == "__main__":
    while True:
        prompt = input("\nask > ")
        print(ask_agent(prompt))
