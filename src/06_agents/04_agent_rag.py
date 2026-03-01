"""
Block 04 - RAG as a Tool
========================
Replaces hardcoded tool functions with a real RAG tool that searches
your documents in ChromaDB.

The agent loop from Block 2 and memory from Block 3 are unchanged.
The only difference is what the tool does when called.

Block 3: get_shipping_info() -> fixed string
Block 4: search_docs(query)  -> relevant chunks from ChromaDB

Flow:
    question -> agent loop -> search_docs tool -> ChromaDB
             -> retrieved chunks -> Ollama -> answer
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# --- Config ----
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
CHROMA_PATH  = "src/rag/docs/chroma_db_rag"
COLLECTION   = "rag_docs"

# --- Load embedder + ChromaDB ----
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print(f"Device:   {device}")
print(f"Chunks:   {collection.count()}")

# --- RAG tool ---- 
# This is the only tool the agent has in Block 4.
# Instead of returning a fixed string, it searches ChromaDB for relevant chunks.

def search_docs(query, top_k=3):
    """
    Search ChromaDB for chunks relevant to the query.

    This is the RAG retrieval step — same as Block 6 of the RAG series
    but wrapped as an agent tool so the loop can call it by name.

    Args:
        query:  the search query (usually the user's question)
        top_k:  number of chunks to retrieve

    Returns:
        a formatted string of the top matching chunks with their sources
    """
    embedding = embedder.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    # Format results as a readable string for the LLM
    chunks = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        score = round(1 - dist, 3)
        chunks.append(f"[score {score}] ({meta['source']}) {doc}")

    return "\n".join(chunks)

# Test the tool directly before wiring it into the agent
print("\nTesting search_docs tool:")
print("─" * 60)

test_queries = [
    "How do I get a refund?",
    "Can I return a sale item?",
]

for q in test_queries:
    print(f"\nQuery: {q}")
    print(search_docs(q))



# --- Tool registry ----
# Only one tool this time - search_docs handles everything.
# The agent decides what query to pass it based on the question.

TOOLS = [
    {
        "name":        "search_docs",
        "description": "Search the company documents to find information about policies, shipping, returns, refunds, or any other company topic.",
        "function":    lambda: None,  # called differently - see run_agent below
    },
]

MAX_STEPS = 4


# --- Agent loop ----

def run_agent(question):
    """
    ReAct loop with a single RAG tool.

    Key difference from Block 2: the tool takes the question as input.
    The agent doesn't just pick the tool name — it also passes the
    question as the search query.

    Args:
        question: the user's question

    Returns:
        answer string
    """
    step_history = []   # retrieved chunks gathered so far
    called_tools = set()

    for step in range(MAX_STEPS):
        remaining_tools = [t for t in TOOLS if t["name"] not in called_tools]

        if not remaining_tools:
            picked = "ANSWER"
        else:
            tool_lines = "\n".join(
                f"- {t['name']}: {t['description']}" for t in remaining_tools
            )
            tool_lines += "\n- ANSWER: Use this when you have enough information to answer the question."

            gathered = ""
            if step_history:
                gathered = "\nWhat you have gathered so far:\n" + "\n".join(
                    f"  {h['result']}" for h in step_history
                )

            pick_prompt = f"""You are an agent. You must pick ONE action from this exact list:

{tool_lines}

These are the ONLY valid responses. Do not pick anything else.
{gathered}

Question: {question}

If the gathered information above is sufficient to answer the question, reply: ANSWER
Otherwise reply with exactly one tool name from the list above.

Your response (one word only):"""

            raw    = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": pick_prompt, "stream": False},
            ).json()["response"]
            picked = raw.strip().strip("'\"")

        print(f"  Step {step + 1}: picked '{picked}'")

        if picked == "ANSWER":
            # Generate final answer from retrieved chunks
            chunks_text = "\n".join(h["result"] for h in step_history)
            answer_prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{chunks_text}

Question: {question}
Answer:"""
            answer = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": answer_prompt, "stream": False},
            ).json()["response"].strip()
            return answer

        # For search_docs, pass the question as the query
        if picked == "search_docs":
            result = search_docs(question)
            step_history.append({"tool": picked, "result": result})
            called_tools.add(picked)

    return "Could not answer within step limit."


# --- Run ----

if __name__ == "__main__":
    questions = [
        "How do I get a refund?",
        "Can I return a sale item?",
        "What payment methods do you accept?",
        "How long does express shipping take?",
    ]

    print("\nAGENT WITH RAG TOOL:")
    print("─" * 60)

    for q in questions:
        print(f"\nQ: {q}")
        answer = run_agent(q)
        print(f"A: {answer}")    
