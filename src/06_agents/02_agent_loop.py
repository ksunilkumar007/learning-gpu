"""
Block 02 - The Agent Loop
=========================
Teaches the ReAct loop: Reason -> Act -> Observe -> Reason -> ...

In Block 1 the LLM always called exactly one tool then answered.
Here the LLM decides at each step whether to call another tool
or stop and return a final answer.

The key addition: a STOP signal called ANSWER.
When the LLM picks ANSWER instead of a tool, the loop ends.

Flow:
    1. Show LLM the question + all tool descriptions
    2. LLM picks a tool OR picks ANSWER
    3. If tool  -> call it, add result to history, go to step 1
    4. If ANSWER -> ask LLM to produce final answer from history
"""

import requests

# --- Config ----
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
MAX_STEPS    = 5  # safety limit — prevent infinite loops


# --- Tools ----

def get_refund_policy():
    """Return the company refund policy."""
    return (
        "Refunds are processed within 5-7 business days. "
        "Contact support@company.com with your order number."
    )

def get_shipping_info():
    """Return shipping timeframes and costs."""
    return (
        "Standard shipping 3-5 days. "
        "Express 1-2 days. "
        "Free shipping on orders over $50."
    )

def get_return_policy():
    """Return the product return policy."""
    return (
        "Items can be returned within 30 days of purchase in original condition. "
        "Sale items are final and cannot be returned. "
        "Start a return at returns.company.com."
    )

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
    {
        "name":        "get_return_policy",
        "description": "Use this when the user asks about returning items or the return policy.",
        "function":    get_return_policy,
    },
]


# --- Helpers ----

def build_tool_list():
    """
    Format tools into a string the LLM can read.
    Includes the special ANSWER signal so the LLM knows how to stop.
    """
    tool_lines = "\n".join(
        f"- {t['name']}: {t['description']}" for t in TOOLS
    )
    # ANSWER is not a real tool — it's a stop signal.
    # When the LLM picks ANSWER it means "I have enough info."
    return tool_lines + "\n- ANSWER: Use this when you have enough information to answer the question."

def call_tool(name):
    """Look up a tool by name and call it. Returns None if name is ANSWER."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["function"]()
    return None


# --- Agent loop ----

def run_agent(question):
    """
    ReAct loop — runs until the LLM picks ANSWER or MAX_STEPS is reached.

    Key change: already-called tools are removed from the tool list each step.
    We enforce no-repeat in code, not by prompting the LLM to remember.

    Args:
        question: the user's question as a plain string

    Returns:
        tuple of (final_answer, steps_taken)
    """
    history      = []
    called_tools = set()  # track which tools have been called

    for step in range(MAX_STEPS):
        # Remove already-called tools so the LLM cannot pick them again
        remaining_tools = [t for t in TOOLS if t["name"] not in called_tools]

        # If all tools have been called, force ANSWER
        if not remaining_tools:
            picked = "ANSWER"
        else:
            # Build tool list from only the remaining available tools
            tool_lines = "\n".join(
                f"- {t['name']}: {t['description']}" for t in remaining_tools
            )
            tool_lines += "\n- ANSWER: Use this when you have enough information to answer the question."

            # Show the LLM what it has already gathered this run
            history_text = ""
            if history:
                history_text = "\nWhat you have gathered so far:\n" + "\n".join(
                    f"  [{h['tool']}]: {h['result']}" for h in history
                )

            pick_prompt = f"""You are an agent. You must pick ONE action from this exact list:

{tool_lines}

These are the ONLY valid responses. Do not pick anything else.
Do not repeat a tool you have already used.

{history_text}

Question: {question}

If the gathered information above is sufficient to answer the question, reply: ANSWER
Otherwise reply with exactly one tool name from the list above.

Your response (one word only):"""

            raw    = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": pick_prompt, "stream": False},
            ).json()["response"]
            # llama3 sometimes wraps the tool name in quotes — strip them
            picked = raw.strip().strip("'\"")
            print(f"  Step {step + 1}: picked '{picked}'")

        # Stop signal — LLM has enough information, generate final answer
        if picked == "ANSWER":
            answer_prompt = f"""Answer the question using only the information below.

{chr(10).join(f'- {h["tool"]}: {h["result"]}' for h in history)}

Question: {question}
Answer:"""
            answer = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": answer_prompt, "stream": False},
            ).json()["response"].strip()
            return answer, step + 1

        # Call the tool, record the result, mark it as used
        result = call_tool(picked)
        if result:
            history.append({"tool": picked, "result": result})
            called_tools.add(picked)  # prevents this tool appearing again next step

    # Safety fallback — should not reach here in normal operation
    return "Could not answer within step limit.", MAX_STEPS

# --- Run ----

if __name__ == "__main__":
    questions = [
        "How long does delivery take?",
        "Can I return a sale item and get a refund?",  # needs 2 tools
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer, steps = run_agent(q)
        print(f"A: {answer}")
        print(f"Steps taken: {steps}")
