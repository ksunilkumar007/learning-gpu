"""
Block 01 - MCP Basics
=====================
MCP (Model Context Protocol) is a standard for exposing tools to LLMs.
The library requires Python 3.10+ so we implement the protocol directly.

MCP is two HTTP endpoints:
    POST /tools/list              -> returns available tool definitions
    POST /tools/call {name, args} -> runs a tool, returns the result

A tool definition has three fields:
    name        -> unique identifier, what the client calls it by
    description -> what the tool does, read by the LLM to decide when to use it
    inputSchema -> JSON Schema describing expected arguments (MCP standard)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
import uvicorn

# --- App ----
app = FastAPI(title="MCP Server")


# --- MCP protocol types ----

class ToolCallRequest(BaseModel):
    name:      str   # tool name to call
    arguments: dict  # tool arguments as key-value pairs


# --- Tool functions ----

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


# --- Tool registry ----
# Each tool has:
#   name        -> called by client to invoke this tool
#   description -> read by LLM to decide when to call it
#   inputSchema -> JSON Schema of arguments (empty object = no args needed)
#   function    -> the Python function to run (not sent to client)

TOOLS = [
    {
        "name":        "get_date",
        "description": "Returns today's date. Use when the user asks what day or date it is.",
        "inputSchema": {
            "type":       "object",
            "properties": {},   # no arguments needed
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


# --- MCP endpoints ----

@app.post("/tools/list")
def list_tools():
    """
    MCP tool discovery - returns all registered tools.
    Strips the internal 'function' key before sending to client.
    Clients must never see the function object - only the schema.
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
    MCP tool execution - finds the tool by name and calls it.
    Returns result wrapped in MCP content format.
    """
    for tool in TOOLS:
        if tool["name"] == req.name:
            if req.arguments:
                result = tool["function"](**req.arguments)
            else:
                result = tool["function"]()
            # MCP response format: content array of typed blocks
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
    """Liveness check - reports how many tools are registered."""
    return {
        "status":   "ok",
        "protocol": "MCP",
        "tools":    len(TOOLS),
    }


# --- Run ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)