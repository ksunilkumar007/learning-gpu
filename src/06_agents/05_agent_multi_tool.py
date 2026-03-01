
""" 
Block 05 - Multiple Tools
=========================
Gives the agent three tools and lets it pick the right one
based on the type of question asked.

Tools:
    search_docs  -> searches ChromaDB for company policy questions
    calculator   -> evaluates math expressions
    get_date     -> returns today's date

The agent loop is identical to Block 4. The tool registry is larger.

Key concept: the agent reasons about WHICH tool fits the question,
not just whether to call a tool at all.
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import re
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import date
import torch

# ---- Config ----
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
CHROMA_PATH  = "src/rag/docs/chroma_db_rag"
COLLECTION   = "rag_docs"
MAX_STEPS    = 5

# --- Load embedder + ChromaDB ----
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print(f"Device: {device}")
print(f"Chunks: {collection.count()}")

# --- Tool functions ----
def search_docs(query, top_k=3):
    """
    Search ChromaDB for chunks relevant to the query.

    Args:
        query:  the search query
        top_k:  number of chunks to retrieve

    Returns:
        formatted string of top matching chunks with scores and sources
    """
    embedding = embedder.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    chunks = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        score = round(1 - dist, 3)
        chunks.append(f"[score:{score}] ({meta['source']}) {doc}")
    return "\n".join(chunks)

def calculator(expression):
    """
    Safely evaluate a math expression and return the result.

    Uses eval() with a restricted namespace - no builtins,
    only basic math operations are allowed.

    Args:
        expression: a math expression as a string e.g. "15 * 84.99 / 100"

    Returns:
        the result as a string, or an error message
    """
    try:
        # Restrict eval to safe math only - no builtins, no imports
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"

def get_date():
    """Return today's date as a readable string."""
    return f"Today's date is {date.today().strftime('%A, %B %d, %Y')}."

# Test all three tools standalone
print("\nTesting tools:")
print(f"  search_docs:  {search_docs('refund policy')[:60]}...")
print(f"  calculator:   {calculator('15 * 84.99 / 100')}")
print(f"  get_date:     {get_date()}")


# --- Tool registry ----
# Three tools - the agent picks the right one based on the question type.
# Descriptions are written to make the choice unambiguous.

TOOLS = [
    {
        "name":        "search_docs",
        "description": "Use this for questions about company policies, shipping, returns, refunds, payments, or anything that requires looking up company information.",
        "function":    "rag",       # special - called with question as arg
    },
    {
        "name":        "calculator",
        "description": "Use this for any math calculation, percentages, totals, or arithmetic.",
        "function":    "math",      # special - called with expression as arg
    },
    {
        "name":        "get_date",
        "description": "Use this when the user asks about today's date or the current day.",
        "function":    get_date,    # no args needed
    },
]


# --- Helpers ----
def extract_expression(question):
    """
    Extract a math expression from a natural language question.
    Uses string replacement rather than regex — more predictable.

    Examples:
        "15% tip on $84.99"  → "0.15 * 84.99"
        "20% off $150"       → "0.20 * 150"
        "what is 42 * 3"     → "42 * 3"

    Args:
        question: natural language math question

    Returns:
        a Python-evaluable expression string, or None if not found
    """
    import re

    q = question.lower()

    # Normalise unicode and word operators
    q = q.replace("×", "*").replace("÷", "/").replace("−", "-")
    q = q.replace(" plus ", " + ").replace(" minus ", " - ")
    q = q.replace(" times ", " * ").replace(" divided by ", " / ")

    # Remove currency symbols and noise words
    q = q.replace("$", "").replace(",", "")

    # Handle "X% tip on Y", "X% of Y", "X% off Y"
    pct_match = re.search(r'(\d+\.?\d*)\s*%[^0-9]*(\d+\.?\d*)', q)
    if pct_match:
        pct   = float(pct_match.group(1)) / 100
        value = pct_match.group(2)
        return f"{pct} * {value}"

    # Fall back: find any sequence of digits and operators
    tokens = re.findall(r'[\d\.]+|[\+\-\*\/\(\)]', q)
    if len(tokens) >= 3:  # need at least: number operator number
        return " ".join(tokens)

    return None

def call_tool(name, question):
    """
    Call the named tool with appropriate arguments.

    Args:
        name:     tool name picked by the LLM
        question: the user's question

    Returns:
        tool result as a string
    """
    if name == "search_docs":
        return search_docs(question)

    elif name == "calculator":
        expression = extract_expression(question)
        if expression is None:
            return "Could not extract a math expression from the question."
        return calculator(expression)

    elif name == "get_date":
        return get_date()

    return "Tool not found"

# --- Agent loop ----

def run_agent(question):
    """
    ReAct loop with three tools.

    The agent picks the tool that matches the question type:
        policy/docs question -> search_docs
        math question        -> calculator
        date question        -> get_date

    Args:
        question: the user's question

    Returns:
        answer string
    """
    step_history = []
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
                    f"  [{h['tool']}]: {h['result']}" for h in step_history
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
            gathered_text = "\n".join(
                f"- [{h['tool']}]: {h['result']}" for h in step_history
            )
            answer_prompt = f"""Answer the question using only the information below.
If the answer is not in the context, say "I don't know".

Information:
{gathered_text}

Question: {question}
Answer:"""
            answer = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": answer_prompt, "stream": False},
            ).json()["response"].strip()
            return answer

        result = call_tool(picked, question)
        step_history.append({"tool": picked, "result": result})
        called_tools.add(picked)

    return "Could not answer within step limit."


# --- Run ----

if __name__ == "__main__":
    questions = [
        "What is the refund policy?",
        "What is 15% tip on a $84.99 order?",
        "What is today's date?",
        "Can I return a sale item?",
        "How much is 20% off a $150 item?",
    ]

    print("\nAGENT WITH MULTIPLE TOOLS:")
    print("─" * 60)

    for q in questions:
        print(f"\nQ: {q}")
        answer = run_agent(q)
        print(f"A: {answer}")
