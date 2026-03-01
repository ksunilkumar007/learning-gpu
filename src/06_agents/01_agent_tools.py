"""
Block 01 — Agent Tools
======================
Teaches the foundation of every agent system: tools.

A tool is a plain Python function wrapped with a name and description
so an LLM can read what it does and decide when to call it.

Flow:
    Step 1: define two plain functions
    Step 2: wrap them as tools with descriptions the LLM can read
    Step 3: LLM reads descriptions, picks the right tool by name
    Step 4: call the tool, feed result back to LLM for final answer

The three-step agent pattern (used in every block going forward):
    pick tool → call tool → answer with result
"""

import requests

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"


# ── Tool functions ────────────────────────────────────────────────────────────
# These are plain Python functions — nothing special about them yet.
# The LLM never calls these directly; your code does, based on what the LLM picks.

def get_refund_policy():
    """Return the company refund policy as a plain string."""
    return (
        "Refunds are processed within 5-7 business days. "
        "Contact support@company.com with your order number."
    )

def get_shipping_info():
    """Return shipping timeframes and cost thresholds as a plain string."""
    return (
        "Standard shipping 3-5 days. "
        "Express 1-2 days. "
        "Free shipping on orders over $50."
    )


# ── Tool registry ─────────────────────────────────────────────────────────────
# Each tool has three fields:
#   name        → what the LLM writes when it wants to call this tool
#   description → what the LLM reads to decide WHEN to call this tool
#   function    → the Python function to actually run
#
# Description style: "Use this when the user asks about X"
# Write it from the user's perspective, not the implementation's.

TOOLS = [
    {
        "name":        "get_refund_policy",
        "description": "Use this when the user asks about refunds, money back, or reimbursement.",
        "function":    get_refund_policy,
    },
    {
        "name":        "get_shipping_info",
        "description": "Use this when the user asks about shipping, delivery, or how long orders take.",
        "function":    get_shipping_info,
    },
]


# ── Helper functions ──────────────────────────────────────────────────────────

def build_tool_list():
    """
    Format all tools into a string the LLM can read.

    Example output:
        - get_refund_policy: Use this when the user asks about refunds...
        - get_shipping_info: Use this when the user asks about shipping...
    """
    return "\n".join(
        f"- {t['name']}: {t['description']}" for t in TOOLS
    )

def call_tool(name):
    """
    Look up a tool by name and call its function.

    Args:
        name: the tool name the LLM returned

    Returns:
        The tool's output string, or an error message if not found.
    """
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["function"]()
    return f"Tool '{name}' not found"

def ask(question):
    """
    Full three-step agent pattern for a single question.

    Step 1 — Pick:   LLM reads tool descriptions + question, returns tool name
    Step 2 — Call:   run the chosen tool function, get its result
    Step 3 — Answer: LLM reads tool result + question, returns final answer

    Args:
        question: the user's question as a plain string

    Returns:
        tuple of (tool_name, tool_result, answer)
    """

    # Step 1: ask the LLM which tool to use
    # The prompt constrains the LLM to reply with ONLY the tool name —
    # no explanation, no punctuation, just the name so we can look it up.
    pick_prompt = f"""You are an assistant with access to these tools:

{build_tool_list()}

Reply with ONLY the tool name to answer this question. No explanation.

Question: {question}
Tool:"""

    tool_name = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": pick_prompt, "stream": False},
    ).json()["response"].strip()

    # Step 2: call the tool — plain Python, no LLM involved here
    tool_result = call_tool(tool_name)

    # Step 3: ask the LLM to answer the question using the tool result
    # We constrain it to only use the provided information so it doesn't
    # hallucinate details not in the tool output.
    answer_prompt = f"""Answer the question using only the information below.

Information: {tool_result}

Question: {question}
Answer:"""

    answer = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": answer_prompt, "stream": False},
    ).json()["response"].strip()

    return tool_name, tool_result, answer


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Print available tools so you can see what the LLM will read
    print("Available tools:\n")
    for tool in TOOLS:
        print(f"  {tool['name']}")
        print(f"  → {tool['description']}\n")

    # Test questions — each should route to the correct tool
    questions = [
        "How do I get my money back?",
        "How long does delivery take?",
        "What is your return policy?",
        "When will my order arrive?",
    ]

    print("─" * 60)
    for q in questions:
        tool_name, tool_result, answer = ask(q)
        print(f"Q:      {q}")
        print(f"Tool:   {tool_name}")
        print(f"Result: {tool_result}")
        print(f"Answer: {answer}")
        print()
