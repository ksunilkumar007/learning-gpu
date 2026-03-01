"""
MCP Server — Production
========================
A production-ready MCP server that exposes RAG + agent capabilities
over the Model Context Protocol.

Endpoints:
    GET  /health          → liveness check, reports device + chunks + tools
    POST /tools/list      → MCP tool discovery
    POST /tools/call      → MCP single tool execution
    POST /agent/ask       → full ReAct agent loop, returns answer + steps + sources

Tools:
    search_docs(query)        → searches ChromaDB for relevant document chunks
    get_date()                → returns today's date
    calculate(expression)     → evaluates a math expression safely

Usage:
    # Start server
    uv run python src/mcp/server.py

    # Health check
    curl http://localhost:8001/health

    # Discover tools
    curl -X POST http://localhost:8001/tools/list

    # Ask the agent
    curl -X POST http://localhost:8001/agent/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "What is the refund policy?"}'
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
PORT         = 8001


# ── Startup — load models once ────────────────────────────────────────────────
# Models are loaded at module level so they are ready before the first request.
# Loading inside a request handler would add 10-30s latency per call.

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print(f"Device:  {device}")
print(f"Chunks:  {collection.count()}")
print(f"Port:    {PORT}")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MCP Server",
    description="RAG + Agent over Model Context Protocol",
    version="1.0.0",
)


# ── Request / response schemas ────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    name:      str    # tool name to call
    arguments: dict   # tool arguments as key-value pairs

class AgentRequest(BaseModel):
    question: str     # the user's question


# ── Tool functions ────────────────────────────────────────────────────────────

def search_docs(query: str, top_k: int = 3):
    """
    Search ChromaDB for document chunks relevant to the query.

    Args:
        query:  search query — usually the user's question verbatim
        top_k:  number of chunks to retrieve (default 3)

    Returns:
        newline-separated string of chunks with similarity scores and sources
    """
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
    """
    Return today's date as a human-readable string.

    Returns:
        e.g. "Today's date is Sunday, March 01, 2026."
    """
    return f"Today's date is {date.today().strftime('%A, %B %d, %Y')}."

def calculate(expression: str):
    """
    Safely evaluate a math expression using eval with no builtins.

    Args:
        expression: a Python math expression e.g. "0.15 * 84.99"

    Returns:
        "expression = result" string, or an error message
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"

def extract_expression(question: str):
    """
    Extract a math expression from a natural language question.
    Uses regex — more reliable than asking the LLM.

    Handles:
        "20% of $150"       -> "0.2 * 150"
        "15% tip on $84.99" -> "0.15 * 84.99"
        "42 * 3"            -> "42 * 3"

    Args:
        question: natural language math question

    Returns:
        a Python-evaluable expression string
    """
    q = question.lower().replace("$", "").replace(",", "")
    q = q.replace("×", "*").replace("÷", "/")

    # Percentage questions — [^0-9]* matches any words between % and number
    pct_match = re.search(r'(\d+\.?\d*)\s*%[^0-9]*(\d+\.?\d*)', q)
    if pct_match:
        pct   = float(pct_match.group(1)) / 100
        value = pct_match.group(2)
        return f"{pct} * {value}"

    # Fall back — extract digits and operators
    tokens = re.findall(r'[\d\.]+|[\+\-\*\/\(\)]', q)
    if len(tokens) >= 3:
        return " ".join(tokens)

    return question


# ── Tool registry ─────────────────────────────────────────────────────────────
# Each tool has name, description, inputSchema (sent to clients)
# and function (internal only — never sent to clients).

TOOLS = [
    {
        "name":        "search_docs",
        "description": "Search company documents for questions about policies, shipping, returns, refunds, or payments.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "The search query — usually the user's question",
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
    ReAct loop — Reason, Act, Observe, repeat until ANSWER.

    Runs entirely on the server. The client receives only the final
    answer, the steps taken, and the document sources used.

    Args:
        question: the user's question

    Returns:
        dict with keys: answer, steps, sources
    """
    step_history = []   # tool results gathered this run
    called_tools = set()
    steps        = []   # audit trail returned to client

    for _ in range(MAX_STEPS):
        # Remove already-called tools — LLM cannot pick what isn't listed
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
            # Build final answer from all gathered tool results
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

            # Extract unique document sources from search_docs results
            sources = list({
                src
                for h in step_history
                if h["tool"] == "search_docs"
                for src in re.findall(r'\(([^)]+\.(?:txt|md))\)', h["result"])
            })

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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — reports device, chunk count, and registered tools."""
    return {
        "status":   "ok",
        "protocol": "MCP",
        "device":   str(device),
        "chunks":   collection.count(),
        "tools":    len(TOOLS),
        "model":    OLLAMA_MODEL,
    }

@app.post("/tools/list")
def list_tools():
    """
    MCP tool discovery.
    Returns all registered tools with names, descriptions, and input schemas.
    The internal function reference is never included.
    """
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
    """
    MCP tool execution — single tool call.
    Used by MCP clients that manage their own reasoning loop.
    """
    for tool in TOOLS:
        if tool["name"] == req.name:
            result = tool["function"](**req.arguments) if req.arguments else tool["function"]()
            return {"content": [{"type": "text", "text": result}]}
    return {"content": [{"type": "text", "text": f"No tool named '{req.name}' registered."}]}

@app.post("/agent/ask")
def agent_ask(req: AgentRequest = Body(...)):
    """
    Agent endpoint — runs the full ReAct loop on the server.

    The client sends one question and receives:
        answer  → the final answer
        steps   → audit trail of tools called
        sources → document files the answer came from
    """
    return run_agent(req.question)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)