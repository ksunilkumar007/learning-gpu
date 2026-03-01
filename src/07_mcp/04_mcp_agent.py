"""
Block 04 — Agent Loop Behind MCP
==================================
The MCP server runs the full ReAct loop internally.
The client sends one question and gets one answer back.
All multi-step reasoning happens on the server.

Block 3: client picks tool -> calls it -> answers
Block 4: client POST /agent/ask -> server loops -> final answer

New endpoint:
    POST /agent/ask {"question": "..."}
    -> server runs ReAct loop
    -> returns {"answer": "...", "steps": [...], "sources": [...]}

The /tools/list and /tools/call endpoints stay — the server is
still a valid MCP tool server AND an agent server.
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Body
from pydantic import BaseModel
from datetime import date
import torch
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
CHROMA_PATH  = "src/rag/docs/chroma_db_rag"
COLLECTION   = "rag_docs"
MAX_STEPS    = 5

# ── Load embedder + ChromaDB ──────────────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print(f"Device:  {device}")
print(f"Chunks:  {collection.count()}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MCP Agent Server")


# ── MCP protocol types ────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    name:      str
    arguments: dict

class AgentRequest(BaseModel):
    question: str


# ── Tool functions ────────────────────────────────────────────────────────────

def search_docs(query: str, top_k: int = 3):
    """Search ChromaDB for chunks relevant to the query."""
    embedding = embedder.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
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

def get_date():
    """Return today's date."""
    return f"Today's date is {date.today().strftime('%A, %B %d, %Y')}."

def calculate(expression: str):
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"

def extract_expression(question: str):
    """
    Extract a math expression from a natural language question.
    Used internally by the agent when it picks the calculate tool.
    """
    q = question.lower().replace("$", "").replace(",", "")
    q = q.replace("×", "*").replace("÷", "/")

    pct_match = re.search(r'(\d+\.?\d*)\s*%[^0-9]*(\d+\.?\d*)', q)
    if pct_match:
        pct   = float(pct_match.group(1)) / 100
        value = pct_match.group(2)
        return f"{pct} * {value}"

    tokens = re.findall(r'[\d\.]+|[\+\-\*\/\(\)]', q)
    if len(tokens) >= 3:
        return " ".join(tokens)

    return question


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name":        "search_docs",
        "description": "Search company documents for questions about policies, shipping, returns, refunds, or payments.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
        "function": search_docs,
    },
    {
        "name":        "get_date",
        "description": "Returns today's date. Use when the user asks what day or date it is.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "function":    get_date,
    },
    {
        "name":        "calculate",
        "description": "Evaluates a math expression. Use for arithmetic, percentages, or calculations.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "expression": {
                    "type":        "string",
                    "description": "A Python math expression e.g. '0.2 * 150'",
                }
            },
            "required": ["expression"],
        },
        "function": calculate,
    },
]


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(question: str):
    """
    ReAct loop — runs entirely on the server.

    The client never sees the individual tool calls — it only receives
    the final answer, steps taken, and sources used.

    Args:
        question: the user's question

    Returns:
        dict with answer, steps list, and sources list
    """
    step_history = []   # tool results gathered this run
    called_tools = set()
    steps        = []   # log of what happened — returned to client

    for step in range(MAX_STEPS):
        remaining = [t for t in TOOLS if t["name"] not in called_tools]

        if not remaining:
            picked = "ANSWER"
        else:
            tool_lines = "\n".join(
                f"- {t['name']}: {t['description']}" for t in remaining
            )
            tool_lines += "\n- ANSWER: Use this when you have enough information to answer."

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

If the gathered information is sufficient to answer the question, reply: ANSWER
Otherwise reply with exactly one tool name from the list above.

Your response (one word only):"""

            raw    = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": pick_prompt, "stream": False},
            ).json()["response"]
            picked = raw.strip().strip("'\"")

        steps.append(picked)

        if picked == "ANSWER":
            # Generate final answer from all gathered tool results
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

            # Collect unique sources from search_docs results
            sources = []
            for h in step_history:
                if h["tool"] == "search_docs":
                    import re as _re
                    found = _re.findall(r'\(([^)]+\.(?:txt|md))\)', h["result"])
                    sources.extend(found)
            sources = list(set(sources))

            return {"answer": answer, "steps": steps, "sources": sources}

        # Call the tool with appropriate arguments
        if picked == "search_docs":
            result = search_docs(question)
        elif picked == "calculate":
            result = calculate(extract_expression(question))
        elif picked == "get_date":
            result = get_date()
        else:
            result = f"Unknown tool: {picked}"

        step_history.append({"tool": picked, "result": result})
        called_tools.add(picked)

    return {
        "answer":  "Could not answer within step limit.",
        "steps":   steps,
        "sources": [],
    }


# ── MCP endpoints ─────────────────────────────────────────────────────────────

@app.post("/tools/list")
def list_tools():
    """MCP tool discovery."""
    return {
        "tools": [
            {
                "name":        t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in TOOLS
        ]
    }

@app.post("/tools/call")
def server_call_tool(req: ToolCallRequest = Body(...)):
    """MCP tool execution — single tool call."""
    for tool in TOOLS:
        if tool["name"] == req.name:
            result = tool["function"](**req.arguments) if req.arguments else tool["function"]()
            return {"content": [{"type": "text", "text": result}]}
    return {"content": [{"type": "text", "text": f"No tool named '{req.name}' registered."}]}

@app.post("/agent/ask")
def agent_ask(req: AgentRequest = Body(...)):
    """
    Agent endpoint — runs the full ReAct loop on the server.
    Client sends a question, gets back answer + steps + sources.
    """
    return run_agent(req.question)

@app.get("/health")
def health():
    return {
        "status":   "ok",
        "protocol": "MCP",
        "tools":    len(TOOLS),
        "chunks":   collection.count(),
        "device":   str(device),
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
