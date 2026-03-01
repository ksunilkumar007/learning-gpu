import requests
import json

OLLAMA_URL = "http://localhost:11434"

def list_models():
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    models = response.json()["models"]
    print("Available models:")
    for m in models:
        print(f"  {m['name']:30s} | {m['size']/1e9:.2f} GB")
    return models

def generate(prompt, model="tinyllama", stream=False):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
    )
    return response.json()["response"]

def chat(messages, model="tinyllama"):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]

if __name__ == "__main__":
    # 1. List models
    print("=== Models ===")
    list_models()

    # 2. Generate
    print("\n=== Generate ===")
    print(generate("What is backpropagation? One sentence."))

    # 3. Chat
    print("\n=== Chat ===")
    messages = [
        {"role": "user", "content": "What is overfitting in machine learning?"}
    ]
    print(chat(messages))

    # 4. Multi-turn chat
    print("\n=== Multi-turn Chat ===")
    history = []
    questions = [
        "What is a neural network?",
        "How does it learn?",
        "What can go wrong?"
    ]
    for q in questions:
        history.append({"role": "user", "content": q})
        answer = chat(history)
        history.append({"role": "assistant", "content": answer})
        print(f"Q: {q}")
        print(f"A: {answer[:200]}")
        print()
