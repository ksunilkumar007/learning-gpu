"""
Block 03 — RAG as an MCP Tool
==============================
Replaces hardcoded tool functions with a real RAG tool that
searches ChromaDB — the same retrieval pipeline from the RAG series.

The MCP server structure is identical to Block 2.
The client (mcp_client.py) is unchanged.
Only the tool changes.

Block 2: get_refund_policy() -> fixed string
Block 3: search_docs(query)  -> live ChromaDB results

Flow:
    client asks question
    -> server search_docs tool queries ChromaDB
    -> returns top matching chunks with scores and sources
    -> client LLM answers from chunks
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, Body
from pydantic import BaseModel
from datetime import date
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "src/rag/docs/chroma_db_rag"
COLLECTION  = "rag_docs"

# ── Load embedder + ChromaDB once at startup ──────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

print(f"Device:  {device}")
print(f"Chunks:  {collection.count()}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MCP RAG Server")


# ── MCP protocol types ────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    name:      str
    arguments: dict


# ── Tool functions ────────────────────────────────────────────────────────────

def search_docs(query: str, top_k: int = 3):
    """
    Search ChromaDB for chunks relevant to the query.

    Same retrieval pipeline as RAG Block 6 — wrapped as an MCP tool
    so any MCP client can call it without knowing about ChromaDB.

    Args:
        query:  the search query — usually the user's question
        top_k:  number of chunks to retrieve

    Returns:
        formatted string of top matching chunks with scores and sources
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
    """Return today's date."""
    return f"Today's date is {date.today().strftime('%A, %B %d, %Y')}."

def calculate(expression: str):
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"


# ── Tool registry ─────────────────────────────────────────────────────────────
# search_docs replaces all the individual policy tools from Block 2.
# One tool handles any company question by searching the actual documents.

TOOLS = [
    {
        "name":        "search_docs",
        "description": "Search company documents for any question about policies, shipping, returns, refunds, payments, or other company topics.",
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


# ── MCP endpoints ─────────────────────────────────────────────────────────────

@app.post("/tools/list")
def list_tools():
    """MCP tool discovery — returns all registered tools."""
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
    """MCP tool execution — calls tool by name with arguments."""
    for tool in TOOLS:
        if tool["name"] == req.name:
            result = tool["function"](**req.arguments) if req.arguments else tool["function"]()
            return {"content": [{"type": "text", "text": result}]}
    return {"content": [{"type": "text", "text": f"No tool named '{req.name}' registered."}]}

@app.get("/health")
def health():
    return {
        "status":   "ok",
        "protocol": "MCP",
        "tools":    len(TOOLS),
        "chunks":   collection.count(),
        "device":   str(device),
    }


# ── Client test ───────────────────────────────────────────────────────────────

def run_client_test():
    """Test the RAG MCP server with the reusable MCPClient."""
    from mcp_client import MCPClient

    # Update MCPClient to handle search_docs arguments
    mclient = MCPClient("http://localhost:8001")
    tools   = mclient.list_tools()

    print("Discovered tools:")
    for t in tools:
        print(f"  {t['name']:20s} → {t['description'][:55]}...")
    print()

    questions = [
        "What is the refund policy?",
        "Can I return a sale item?",
        "What payment methods do you accept?",
        "How long does express shipping take?",
        "What is 15% tip on $84.99?",
        "What is today's date?",
    ]

    print("─" * 60)
    for q in questions:
        # Build arguments based on tool picked
        mclient.list_tools()
        tool_name = mclient.pick_tool(q)

        arguments = {}
        if tool_name == "search_docs":
            arguments = {"query": q}
        elif tool_name == "calculate":
            arguments = {"expression": mclient._extract_expression(q)}

        result = mclient.call_tool(tool_name, arguments)
        answer = mclient.answer(q, result)

        print(f"Q:    {q}")
        print(f"Tool: {tool_name}")
        print(f"A:    {answer}")
        print()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--client" in sys.argv:
        run_client_test()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8001)