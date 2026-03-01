"""
Block 02 — MCP Tools + Client
==============================
Block 1 proved the protocol works with curl.
Block 2 adds more tools and builds a Python MCP client that:
    1. Connects to the server
    2. Discovers available tools automatically
    3. Picks the right tool for a question
    4. Calls it and returns the result

This is exactly how Claude Desktop and Cursor use MCP servers.

Flow:
    client.list_tools()        -> discover what's available
    client.pick_tool(question) -> LLM decides which tool fits
    client.call_tool(name)     -> execute and get result
"""

from fastapi import FastAPI, Body
from pydantic import BaseModel
from datetime import date
import uvicorn

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MCP Server — Block 02")


# ── MCP protocol types ────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    name:      str
    arguments: dict


# ── Tool functions ────────────────────────────────────────────────────────────

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

def get_refund_policy():
    """Return the company refund policy."""
    return (
        "Refunds are processed within 5-7 business days. "
        "Contact support@company.com with your order number. "
        "Requests must be submitted within 60 days of purchase."
    )

def get_shipping_info():
    """Return shipping timeframes and costs."""
    return (
        "Standard shipping 3-5 days. "
        "Express 1-2 days. "
        "Free shipping on orders over $50."
    )


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
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
                    "description": "A Python math expression e.g. '15 * 84.99 / 100'",
                }
            },
            "required": ["expression"],
        },
        "function": calculate,
    },
    {
        "name":        "get_refund_policy",
        "description": "Returns the company refund policy. Use when the user asks about refunds or money back.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "function":    get_refund_policy,
    },
    {
        "name":        "get_shipping_info",
        "description": "Returns shipping timeframes and costs. Use when the user asks about delivery or shipping.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "function":    get_shipping_info,
    },
]


# ── MCP endpoints ─────────────────────────────────────────────────────────────

@app.post("/tools/list")
def list_tools():
    """MCP tool discovery — returns all registered tools without function refs."""
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
    return {"status": "ok", "protocol": "MCP", "tools": len(TOOLS)}


# ── MCP Client test ───────────────────────────────────────────────────────────

def run_client_test():
    """Test the MCP client against the running server."""
    from mcp_client import MCPClient

    client = MCPClient("http://localhost:8001")

    # Step 1 — discover tools
    tools = client.list_tools()
    print("Discovered tools:")
    for t in tools:
        print(f"  {t['name']:25s} → {t['description'][:50]}...")
    print()

    # Step 2 — ask questions
    questions = [
        "What is today's date?",
        "How do I get a refund?",
        "How long does shipping take?",
        "What is 20% of $150?",
    ]

    print("─" * 60)
    for q in questions:
        tool_name, tool_result, answer = client.ask(q)
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