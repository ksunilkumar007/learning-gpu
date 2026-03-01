"""
Block 03 - Agent Memory
=======================
Teaches why memory matters and how to add it.

Without memory, every question starts fresh - the agent cannot
handle follow-up questions or build on previous answers.

With memory, the agent carries a conversation history that grows
with each question, giving the LLM full context at every step.

Flow:
    No memory:   question -> tools -> answer -> forget
    With memory: question -> history + tools -> answer -> remember
"""

import requests

# --- Config ----
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# --- Demo ----

def ask_no_memory(question):
    """Ask a question with no memory - each call is completely isolated."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": question, "stream": False},
    )
    return response.json()["response"].strip()

print("WITHOUT MEMORY:")
print(f"Q: How long does shipping take?")
print(f"A: {ask_no_memory('How long does shipping take? Answer in one sentence.')}\n")

print(f"Q: What about express?")
print(f"A: {ask_no_memory('What about express? Answer in one sentence.')}")
print("  ↑ LLM has no idea what 'express' refers to\n")

# --- Demo - with memory ----

def ask_with_memory(question, history):
    """
    Ask a question with full conversation history.

    The history is a list of {"role": "user"/"assistant", "content": "..."}
    dicts. We format it into a single prompt so the LLM sees the full
    conversation before answering the new question.

    Args:
        question: the new question to ask
        history:  list of previous turns, modified in place

    Returns:
        the assistant's answer as a string
    """
    # Add the new question to history
    history.append({"role": "user", "content": question})

    # Format the full conversation as a prompt
    # Each turn is labelled so the LLM knows who said what
    conversation = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history
    )
    prompt = f"{conversation}\nAssistant:"

    answer = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    ).json()["response"].strip()

    # Store the answer so the next question can see it
    history.append({"role": "assistant", "content": answer})

    return answer

print("WITH MEMORY:")
history = []  # shared across all questions in this conversation

questions = [
    "How long does shipping take? Answer in one sentence.",
    "What about express?",
    "Is there free shipping?",
]

for q in questions:
    answer = ask_with_memory(q, history)
    print(f"Q: {q}")
    print(f"A: {answer}\n")

print(f"History length: {len(history)} turns")

# ── Memory + tool loop ────────────────────────────────────────────────────────

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

MAX_STEPS = 5

def call_tool(name):
    """Look up a tool by name and call it. Returns None if not found."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["function"]()
    return None

def run_agent_with_memory(question, conversation_history):
    """
    ReAct loop with conversation memory.

    Two kinds of history:
        conversation_history -> grows across questions (persists between calls)
        step_history         -> tool results gathered within this question only

    The LLM sees both when picking tools and when generating the final answer.

    Args:
        question:             the current question
        conversation_history: list of previous Q&A turns, modified in place

    Returns:
        answer string
    """
    # Format previous conversation so the LLM has full context
    prior_context = ""
    if conversation_history:
        prior_context = "Previous conversation:\n" + "\n".join(
            f"  {t['role'].capitalize()}: {t['content']}"
            for t in conversation_history
        ) + "\n"

    # step_history holds tool results for THIS question only
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

            # Show gathered tool results for this question
            gathered = ""
            if step_history:
                gathered = "\nWhat you have gathered so far:\n" + "\n".join(
                    f"  [{h['tool']}]: {h['result']}" for h in step_history
                )

            pick_prompt = f"""You are an agent. You must pick ONE action from this exact list:

{tool_lines}

These are the ONLY valid responses. Do not pick anything else.
Do not repeat a tool you have already used.

{prior_context}{gathered}

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
            answer_prompt = f"""Answer the question using only the information below.

{prior_context}
Tool results:
{chr(10).join(f'- {h["tool"]}: {h["result"]}' for h in step_history)}

Question: {question}
Answer:"""

            answer = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": answer_prompt, "stream": False},
            ).json()["response"].strip()

            # Save this Q&A to conversation history for future questions
            conversation_history.append({"role": "user",      "content": question})
            conversation_history.append({"role": "assistant", "content": answer})

            return answer

        result = call_tool(picked)
        if result:
            step_history.append({"tool": picked, "result": result})
            called_tools.add(picked)

    return "Could not answer within step limit."


print("\nAGENT WITH MEMORY:")
print("─" * 60)

# conversation_history persists across all questions
conversation_history = []

questions = [
    "How long does shipping take?",
    "What about express?",              # follow-up - needs memory
    "Can I return something I bought?", # new topic
    "And if it was on sale?",           # follow-up - needs memory
]

for q in questions:
    print(f"\nQ: {q}")
    answer = run_agent_with_memory(q, conversation_history)
    print(f"A: {answer}")

print(f"\nConversation history: {len(conversation_history)} turns")
