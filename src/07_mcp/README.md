# MCP — Model Context Protocol
> RHEL 9 · NVIDIA L4 · Ollama · ChromaDB · FastAPI

---

## What is MCP?

MCP is a standard protocol that lets LLMs discover and call tools
from any MCP-compatible server. Instead of hardcoding tools inside
your agent, you expose them over a standard interface that any client
can connect to.

```
Without MCP:
  Tools live inside your Python script
  Only your script can call them
  Every new client needs custom integration

With MCP:
  Tools are exposed over a standard protocol
  Claude Desktop, Cursor, any MCP client can call them
  One server, many clients
```

---

## How MCP fits into what you've built

```
RAG series    → built retrieval + generation pipeline
Agents series → built tool use + decision loop
MCP series    → expose all of that as a standard protocol

Your RAG API (07_rag_api.py) is already HTTP.
MCP wraps it so LLM clients can discover and call it automatically.
```

---

## The MCP Flow

```
MCP Client (Claude Desktop, Cursor, curl)
        ↓
    connect to MCP server
        ↓
    list_tools()  → server returns tool names + descriptions
        ↓
    call_tool(name, args) → server runs the tool, returns result
        ↓
    LLM uses result to answer
```

---

## Environment

```
Python:    3.9
GPU:       NVIDIA L4 (23GB VRAM)
CUDA:      12.4
Ollama:    running via Podman on port 11434
Model:     llama3
ChromaDB:  src/rag/docs/chroma_db_rag (11 chunks)
```

---

## Setup

```bash
cd ~/learning-gpu

# Create MCP folder
mkdir -p src/mcp

# Install MCP library
uv add mcp

# Ollama must be running
podman start ollama
```

---

## Dependencies

```toml
[project]
dependencies = [
    "mcp",                    # MCP server + client library
    "torch",
    "sentence-transformers",
    "chromadb",
    "requests",
    "fastapi",
    "uvicorn",
]
```

---

## Project Structure

```
src/mcp/
├── 01_mcp_basics.py     # what MCP is, start a server, inspect protocol
├── 02_mcp_tools.py      # register tools, client discovers + calls them
├── 03_mcp_rag.py        # RAG as an MCP tool
├── 04_mcp_agent.py      # full agent loop behind MCP
└── server.py            # production server — everything wired up
```

---

## Key Concepts

### Tool Registration
```python
# MCP tools are declared with a decorator
@server.tool()
def get_refund_policy() -> str:
    """Returns the company refund policy."""
    return "Refunds processed within 5-7 days..."
```

### Tool Discovery
```
Client connects → asks "what tools do you have?"
Server replies  → list of tool names + descriptions + input schemas
Client picks    → calls the right tool with arguments
```

### MCP vs raw FastAPI
```
FastAPI:  you define the routes, clients must know them in advance
MCP:      clients discover tools automatically at runtime
          standard schema means any MCP client works out of the box
```

---

## Results Summary

| Block | Script | What it does |
|-------|--------|--------------|
| 01 | `01_mcp_basics.py` | Start a server, inspect protocol messages |
| 02 | `02_mcp_tools.py`  | Register tools, client discovers + calls them |
| 03 | `03_mcp_rag.py`    | RAG as an MCP tool |
| 04 | `04_mcp_agent.py`  | Full agent loop behind MCP |
| —  | `server.py`        | Production server, everything wired up |

---
## Step 01 — MCP Basics

> **NOTE:** The `mcp` library requires Python 3.10+. This project runs
> Python 3.9 so we implement the protocol directly. MCP is just two
> HTTP endpoints — implementing it manually means you understand exactly
> what the protocol does rather than relying on library magic.
>
> ```
> MCP is two endpoints:
>   POST /tools/list              → what tools exist?
>   POST /tools/call {name, args} → run this tool
> ```
>
> Built in 3 steps:
> ```
> Step 1: Bare server — health, empty tool list, call returns not found
> Step 2: Register first tool with no arguments (get_date)
> Step 3: Register tool with arguments (calculate) + update call_tool
> ```

### The MCP tool definition

Every tool has exactly three fields the client sees:

```python
{
    "name":        "calculate",           # how the client calls it
    "description": "Evaluates a math...", # what the LLM reads to decide when to use it
    "inputSchema": {                      # JSON Schema — what arguments it takes
        "type":       "object",
        "properties": {
            "expression": {
                "type":        "string",
                "description": "A Python math expression e.g. '15 * 84.99 / 100'",
            }
        },
        "required": ["expression"],       # client must provide this
    },
}
```

The `function` key is internal — never sent to the client.

### Step 1 — Protocol verified

```
GET  /health      → {"status": "ok", "protocol": "MCP", "tools": 0}
POST /tools/list  → {"tools": []}
POST /tools/call  → {"content": [{"type": "text", "text": "No tool named..."}]}
```

### Step 2 — First tool registered

```
POST /tools/list  → {"tools": [{"name": "get_date", "inputSchema": {...}}]}
POST /tools/call {"name": "get_date", "arguments": {}}
  → {"content": [{"type": "text", "text": "Today's date is Sunday, March 01, 2026."}]}
```

### Step 3 — Tool with arguments

```
POST /tools/call {"name": "calculate", "arguments": {"expression": "15 * 84.99 / 100"}}
  → {"content": [{"type": "text", "text": "15 * 84.99 / 100 = 12.7485"}]}
```

Key change in `call_tool` — pass arguments to functions that need them:

```python
if req.arguments:
    result = tool["function"](**req.arguments)
else:
    result = tool["function"]()
```

### src/mcp/01_mcp_basics.py

```python
"""
Block 01 — MCP Basics
=====================
MCP (Model Context Protocol) is a standard for exposing tools to LLMs.
The library requires Python 3.10+ so we implement the protocol directly.

MCP is two HTTP endpoints:
    POST /tools/list              → returns available tool definitions
    POST /tools/call {name, args} → runs a tool, returns the result

A tool definition has three fields:
    name        → unique identifier, what the client calls it by
    description → what the LLM reads to decide when to use it
    inputSchema → JSON Schema describing expected arguments (MCP standard)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
import uvicorn

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MCP Server")


# ── MCP protocol types ────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    name:      str   # tool name to call
    arguments: dict  # tool arguments as key-value pairs


# ── Tool functions ────────────────────────────────────────────────────────────

def get_date():
    """Return today's date as a readable string."""
    return f"Today's date is {date.today().strftime('%A, %B %d, %Y')}."

def calculate(expression: str):
    """
    Safely evaluate a math expression.

    Args:
        expression: a Python math expression e.g. "15 * 84.99 / 100"

    Returns:
        the result as a string
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"


# ── Tool registry ─────────────────────────────────────────────────────────────
# Each tool has:
#   name        → called by client to invoke this tool
#   description → read by LLM to decide when to call it
#   inputSchema → JSON Schema of arguments (empty = no args needed)
#   function    → the Python function to run (NOT sent to client)

TOOLS = [
    {
        "name":        "get_date",
        "description": "Returns today's date. Use when the user asks what day or date it is.",
        "inputSchema": {
            "type":       "object",
            "properties": {},
            "required":   [],
        },
        "function": get_date,
    },
    {
        "name":        "calculate",
        "description": "Evaluates a math expression. Use for any arithmetic, percentages, or calculations.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "expression": {
                    "type":        "string",
                    "description": "A Python math expression e.g. '15 * 84.99 / 100'",
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
    """
    MCP tool discovery — returns all registered tools.
    Strips the internal 'function' key — clients never see it.
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
def call_tool(req: ToolCallRequest):
    """
    MCP tool execution — finds the tool by name and calls it.
    Unpacks req.arguments as kwargs for tools that need them.
    """
    for tool in TOOLS:
        if tool["name"] == req.name:
            if req.arguments:
                result = tool["function"](**req.arguments)
            else:
                result = tool["function"]()
            return {
                "content": [
                    {"type": "text", "text": result}
                ]
            }
    return {
        "content": [
            {"type": "text", "text": f"No tool named '{req.name}' registered."}
        ]
    }

@app.get("/health")
def health():
    """Liveness check — reports registered tool count."""
    return {
        "status":   "ok",
        "protocol": "MCP",
        "tools":    len(TOOLS),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

```
GET  /health     → {"status": "ok", "protocol": "MCP", "tools": 2}

POST /tools/list →
{
    "tools": [
        {"name": "get_date",  "inputSchema": {"properties": {}, "required": []}},
        {"name": "calculate", "inputSchema": {"properties": {"expression": {...}}, "required": ["expression"]}}
    ]
}

POST /tools/call {"name": "calculate", "arguments": {"expression": "15 * 84.99 / 100"}}
  → {"content": [{"type": "text", "text": "15 * 84.99 / 100 = 12.7485"}]}
```

> **Key insight:** MCP is not magic — it is two HTTP endpoints and a
> JSON schema convention. Any client that speaks HTTP can use it.
> The `inputSchema` field is what makes it self-describing:
> a client can discover what arguments a tool needs without any
> prior knowledge of the server.
>
> ```
> /tools/list → self-describing tool catalogue
> /tools/call → standard execution interface
> Both together → any MCP client works with any MCP server
> ```

---
## Step 02 — MCP Tools + Client

> **NOTE:** Block 1 proved the protocol works with curl.
> Block 2 adds more tools and builds a Python client that discovers
> and calls them automatically — the way Claude Desktop and Cursor do.
>
> ```
> Block 1: curl → MCP server → result           (manual)
> Block 2: Python client → discover → pick → call → result  (automatic)
> ```
>
> Built in 2 steps:
> ```
> Step 1: Register 4 tools on the server
> Step 2: Build MCPClient — list, pick, call, answer
> ```

### Key fix — Body(...) required for POST endpoints

FastAPI needs an explicit `Body(...)` annotation on POST endpoints
that receive a JSON body. Without it, FastAPI reads from query params
and returns 422.

```python
# Wrong — FastAPI reads from query string
@app.post("/tools/call")
def server_call_tool(req: ToolCallRequest):

# Correct — FastAPI reads from request body
from fastapi import Body
@app.post("/tools/call")
def server_call_tool(req: ToolCallRequest = Body(...)):
```

### Key fix — calculate needs arguments extracted in the client

The client picks the tool by name but must also build the arguments
dict before calling. For `calculate`, the expression is extracted
from the question using regex — not by asking the LLM.

```python
arguments = {}
if tool_name == "calculate":
    arguments = {"expression": self._extract_expression(question)}

tool_result = self.call_tool(tool_name, arguments)
```

### mcp_client.py

The client is a reusable class used across all MCP blocks:

```python
client = MCPClient("http://localhost:8001")

# Discover
tools = client.list_tools()

# Full flow — one call does everything
tool_name, tool_result, answer = client.ask("How do I get a refund?")
```

### Final output

```
Discovered tools:
  get_date                  → Returns today's date...
  calculate                 → Evaluates a math expression...
  get_refund_policy         → Returns the company refund policy...
  get_shipping_info         → Returns shipping timeframes...

Q:    What is today's date?
Tool: get_date
A:    Today's date is Sunday, March 01, 2026.

Q:    How do I get a refund?
Tool: get_refund_policy
A:    Contact support@company.com with your order number within 60 days.

Q:    How long does shipping take?
Tool: get_shipping_info
A:    Standard shipping 3-5 days. Express 1-2 days.

Q:    What is 20% of $150?
Tool: calculate
A:    20% of $150 is $30.
```

> **Key insight:** The client discovers tools at runtime — it never
> hardcodes tool names. Add a new tool to the server and the client
> picks it up automatically on the next call to list_tools().
>
> ```
> server owns:  tool definitions, tool functions, inputSchema
> client owns:  discovery, LLM routing, argument extraction, answering
> ```

---
## Step 03 — RAG as an MCP Tool

> **NOTE:** Block 2 had hardcoded tools returning fixed strings.
> Block 3 replaces them with a single `search_docs` tool that queries
> your actual ChromaDB — the same retrieval pipeline from the RAG series.
>
> ```
> Block 2: get_refund_policy() → fixed string   (4 separate tools)
> Block 3: search_docs(query)  → ChromaDB results (1 tool, any question)
> ```
>
> The MCP server structure is identical to Block 2.
> The client is unchanged. Only the tool changes.
>
> Built in 2 steps:
> ```
> Step 1: Verify ChromaDB loads — device + chunk count
> Step 2: Add search_docs tool + full server + client test
> ```

### Key change — one RAG tool replaces many hardcoded tools

```python
# Block 2: one tool per topic
TOOLS = [
    {"name": "get_refund_policy",  ...},
    {"name": "get_shipping_info",  ...},
    {"name": "get_return_policy",  ...},
    ...
]

# Block 3: one tool for everything in documents
TOOLS = [
    {"name": "search_docs", ...},  # handles any company question
    {"name": "get_date",    ...},  # non-document question
    {"name": "calculate",   ...},  # non-document question
]
```

### search_docs tool definition

```python
{
    "name":        "search_docs",
    "description": "Search company documents for any question about policies,
                    shipping, returns, refunds, payments, or other company topics.",
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
}
```

### src/mcp/03_mcp_rag.py output

```
Device:  cuda
Chunks:  11

Discovered tools:
  search_docs          → Search company documents for any question...
  get_date             → Returns today's date...
  calculate            → Evaluates a math expression...

Q:    What is the refund policy?
Tool: search_docs
A:    Refunds are processed within 5-7 business days. Refund requests
      must be submitted within 60 days. Digital products non-refundable.

Q:    Can I return a sale item?
Tool: search_docs
A:    No, sale items are final sale and cannot be returned or exchanged.

Q:    What payment methods do you accept?
Tool: search_docs
A:    Visa, Mastercard, Amex, PayPal, and Apple Pay.

Q:    How long does express shipping take?
Tool: search_docs
A:    Express shipping takes 1-2 business days.

Q:    What is 15% tip on $84.99?
Tool: calculate
A:    $12.75 (approximately)

Q:    What is today's date?
Tool: get_date
A:    Sunday, March 01, 2026.
```

> **Key insight:** One `search_docs` tool now handles any company
> question regardless of topic. As you add more documents to ChromaDB
> the tool gets more capable without any code changes.
>
> ```
> Block 2: N tools for N topics    → doesn't scale
> Block 3: 1 RAG tool for all docs → scales to any number of documents
> ```
>
> In Block 4 the full agent loop goes behind MCP — the server handles
> multi-step reasoning, not just single tool calls.

---
## Step 04 — Agent Loop Behind MCP

> **NOTE:** Block 3 had a thin server — the client picked tools and
> called them one at a time. Block 4 moves the full ReAct loop into
> the server. The client sends one question and gets one answer back.
>
> ```
> Block 3: client picks tool → calls it → answers        (client is smart)
> Block 4: client POST /agent/ask → server loops → answer (server is smart)
> ```
>
> Built in 2 steps:
> ```
> Step 1: Load ChromaDB + bare server — verify device + chunks
> Step 2: Add tools, agent loop, /agent/ask endpoint
> ```

### New endpoint — /agent/ask

```
POST /agent/ask {"question": "..."}
→ server runs full ReAct loop internally
→ returns {"answer": "...", "steps": [...], "sources": [...]}
```

The client never sees individual tool calls. The server handles all
multi-step reasoning and returns one clean response.

### The server now has two interfaces

```
/tools/list    → MCP tool discovery (any MCP client)
/tools/call    → MCP single tool execution (any MCP client)
/agent/ask     → full agent reasoning (smart clients that want one answer)
/health        → liveness + stats
```

### What the client receives

```json
{
    "answer":  "No. Sale items are final and cannot be returned.",
    "steps":   ["search_docs", "get_date", "calculate", "ANSWER"],
    "sources": ["faq.md", "policy.txt"]
}
```

The client gets answer + audit trail (steps) + provenance (sources).
It never knew the server ran 4 steps to produce that answer.

### Final output

```bash
curl -s -X POST http://localhost:8001/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'

{
    "answer": "Refunds are processed within 5-7 business days.
               Requests must be submitted within 60 days.
               Digital products are non-refundable once downloaded.",
    "steps":   ["search_docs", "get_date", "calculate", "ANSWER"],
    "sources": ["policy.txt"]
}

curl -s -X POST http://localhost:8001/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I return a sale item and get a refund?"}'

{
    "answer":  "No. Sale items are final sale and cannot be returned.",
    "steps":   ["search_docs", "get_date", "calculate", "ANSWER"],
    "sources": ["faq.md", "policy.txt"]
}
```

> **Key insight:** The server over-gathers — llama3 calls all three
> tools before picking ANSWER even for simple questions. This is the
> same llama3 behaviour from the Agents series. Answers are correct
> either way. A larger model stops earlier.
>
> The important thing is that the client is completely unaffected by
> how many steps the server takes:
>
> ```
> client sends:   {"question": "..."}
> client receives: {"answer": "...", "steps": [...], "sources": [...]}
> server complexity is invisible to the client
> ```

---
## server.py — Production Server

> All four blocks built the pieces. `server.py` wires them into one
> clean, production-ready file.
>
> ```
> 01_mcp_basics.py  → protocol
> 02_mcp_tools.py   → tool registry
> 03_mcp_rag.py     → RAG tool
> 04_mcp_agent.py   → agent loop
> server.py         → all of the above, clean, documented, deployable
> ```

### What changed from Block 4

```
No functional changes — same tools, same loop, same endpoints.
Changes are structural:
  - module docstring with full usage instructions
  - PORT extracted to config section
  - inline comments explain every decision
  - sources extraction uses a set comprehension instead of a loop
  - ready to deploy as-is
```

### Endpoints

```
GET  /health       → liveness, reports device + chunks + tools + model
POST /tools/list   → MCP tool discovery — any MCP client
POST /tools/call   → MCP single tool execution — any MCP client
POST /agent/ask    → full ReAct loop — smart clients wanting one answer
```

### Final output

```bash
curl http://localhost:8001/health
{
    "status": "ok", "protocol": "MCP",
    "device": "cuda", "chunks": 11, "tools": 3, "model": "llama3"
}

curl -X POST http://localhost:8001/agent/ask \
     -d '{"question": "Can I return a sale item?"}'
{
    "answer":  "No. Sale items are final sale and cannot be returned.",
    "steps":   ["search_docs", "get_date", "ANSWER"],
    "sources": ["policy.txt", "faq.md"]
}

curl -X POST http://localhost:8001/agent/ask \
     -d '{"question": "What is 20% of $85?"}'
{
    "answer":  "17.0",
    "steps":   ["calculate", "search_docs", "get_date", "ANSWER"],
    "sources": ["policy.txt", "faq.md"]
}

curl -X POST http://localhost:8001/agent/ask \
     -d '{"question": "What is today date?"}'
{
    "answer":  "Today's date is Sunday, March 01, 2026.",
    "steps":   ["get_date", "ANSWER"],
    "sources": []
}
```

---

## MCP Complete

```
01 → protocol    → two endpoints, JSON schema, tool discovery
02 → tools       → register tools, Python client, LLM routing
03 → RAG tool    → ChromaDB replaces hardcoded tool functions
04 → agent loop  → full ReAct loop runs on server, client gets one answer
server.py        → production-ready, all pieces wired together
```

### What you built

```
Any HTTP client can now:
    GET  /health       → check the server is alive
    POST /tools/list   → discover what tools exist and their schemas
    POST /tools/call   → call any tool directly
    POST /agent/ask    → ask a question, get answer + steps + sources
```

### The full stack

```
Documents (policy.txt, faq.md)
    ↓ chunked + embedded
ChromaDB (11 chunks, cosine similarity)
    ↓ retrieved by search_docs tool
MCP Server (FastAPI, port 8001)
    ├── /tools/list  → tool catalogue
    ├── /tools/call  → single tool execution
    └── /agent/ask   → ReAct loop → answer + steps + sources
        ↑
    Ollama (llama3, port 11434, NVIDIA L4 GPU)
```

### Key lessons from the MCP series

```
1. MCP is just HTTP + JSON schema — two endpoints, not magic
2. Body(...) is required for FastAPI POST endpoints with JSON bodies
3. Tool descriptions matter — the LLM reads them to decide what to call
4. Keep argument extraction in code, not in the LLM
5. The server can be both an MCP tool server AND an agent server
```

---