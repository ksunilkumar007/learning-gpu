# Agents — Tool Use and Decision Making
> RHEL 9 · NVIDIA L4 · PyTorch · Ollama · FastAPI

---

## What is an Agent?

RAG answers a question. An agent **completes a task** — it decides what
to do, does it, looks at the result, and decides what to do next.

```
RAG:
  User: "What's the refund policy?"
  System: retrieve → generate → answer. Done.

Agent:
  User: "Find our refund policy, summarize it, and draft a customer email"
  Agent: → call RAG tool       → got policy text
         → call summarize tool → got summary
         → call draft tool     → got email
         → return final result
```

The LLM is the brain. Tools are its hands.

---

## The Agent Loop

```
User gives task
      ↓
LLM thinks: "what tool should I call?"
      ↓
Call tool → get result
      ↓
LLM thinks: "do I have enough to answer?"
      ├── No  → call another tool
      └── Yes → return final answer
```

This loop is called **ReAct** — Reason + Act. The LLM reasons about
what to do, acts by calling a tool, observes the result, and repeats.

---

## Environment

```
Python:    3.9
GPU:       NVIDIA L4 (23GB VRAM)
CUDA:      12.4
PyTorch:   2.6.0+cu124
Ollama:    running via Podman on port 11434
Model:     llama3
```

---

## Setup

```bash
cd ~/learning-gpu

# Create agents folder
mkdir -p src/agents

# Ollama must be running
podman start ollama
curl http://localhost:11434/api/tags
```

---

## Project Structure

```
src/agents/
├── 01_agent_tools.py      # define tools, LLM picks and calls them
├── 02_agent_loop.py       # LLM loops until it has a final answer
├── 03_agent_memory.py     # agent remembers across steps
├── 04_agent_rag.py        # RAG becomes one of the agent's tools
└── 05_agent_multi_tool.py # agent uses RAG + calculator + search
```

---

## Key Concepts

### Tool
```python
# A tool is: name + description + function
{
    "name":        "get_refund_policy",
    "description": "Use this when the user asks about refunds.",
    "function":    get_refund_policy,   # plain Python function
}
```

### Tool Selection
```
LLM reads all tool descriptions + user question
→ replies with the name of the tool to call
→ you call it and return the result
```

### ReAct Loop
```
Reason → Act → Observe → Reason → Act → Observe → ...→ Answer
```

### Three-Step Pattern
```
Step 1: LLM picks a tool     (reads descriptions + question)
Step 2: Call the tool        (plain Python function call)
Step 3: LLM answers          (reads tool result + question)
```

---

## Results Summary

| Block | Script | What it does |
|-------|--------|--------------|
| 01 | `01_agent_tools.py` | Define tools, LLM picks and calls them |
| 02 | `02_agent_loop.py`  | LLM loops until it has a final answer |
| 03 | `03_agent_memory.py` | Agent remembers across steps |
| 04 | `04_agent_rag.py` | RAG as an agent tool |
| 05 | `05_agent_multi_tool.py` | Multiple tools, agent decides |

---

## Step 01 — Agent Tools

> **NOTE:** Before an agent can decide what to call, you need tools it
> can call. A tool is just a Python function with a name and description
> the LLM can read.
>
> ```
> Block 1: define tools → call them manually → LLM picks → LLM answers
> Block 2: LLM loops across multiple tools until task is complete
> ```
>
> Built in 4 steps:
> ```
> Step 1: Two plain functions
> Step 2: Wrap them as tools with descriptions
> Step 3: LLM reads descriptions, picks the right tool
> Step 4: Call the tool, feed result back to LLM for final answer
> ```

### Step 1 — Two plain tools

```python
def get_refund_policy():
    """Returns the company refund policy."""
    return "Refunds are processed within 5-7 business days. Contact support@company.com with your order number."

def get_shipping_info():
    """Returns shipping timeframes and costs."""
    return "Standard shipping 3-5 days. Express 1-2 days. Free shipping on orders over $50."

print(get_refund_policy())
print(get_shipping_info())
```

```
Refunds are processed within 5-7 business days. Contact support@company.com with your order number.
Standard shipping 3-5 days. Express 1-2 days. Free shipping on orders over $50.
```

### Step 2 — Wrap as tools with descriptions

```python
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
```

```
Available tools:

  get_refund_policy
  → Use this when the user asks about refunds, money back, or reimbursement.

  get_shipping_info
  → Use this when the user asks about shipping, delivery, or how long orders take.
```

> **Key insight:** The `description` field is what the LLM reads to
> decide which tool to call. Write it from the user's perspective —
> "Use this when the user asks about..."

### Step 3 — LLM picks the tool

```python
def pick_tool(question):
    prompt = f"""You are an assistant with access to these tools:

{build_tool_list()}

Reply with ONLY the tool name to answer this question. No explanation.

Question: {question}
Tool:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )
    return response.json()["response"].strip()
```

```
Q: How do I get my money back?
→ get_refund_policy

Q: How long does delivery take?
→ get_shipping_info

Q: What is your return policy?
→ get_refund_policy

Q: When will my order arrive?
→ get_shipping_info
```

> **Key insight:** The LLM is routing by meaning, not keywords.
> "money back" → refund tool. "order arrive" → shipping tool.
> 4 for 4 correct routing.

### Step 4 — Call the tool and answer

### src/agents/01_agent_tools.py

```python
import requests

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

def get_refund_policy():
    return "Refunds are processed within 5-7 business days. Contact support@company.com with your order number."

def get_shipping_info():
    return "Standard shipping 3-5 days. Express 1-2 days. Free shipping on orders over $50."

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

def build_tool_list():
    return "\n".join(
        f"- {t['name']}: {t['description']}" for t in TOOLS
    )

def call_tool(name):
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["function"]()
    return "Tool not found"

def ask(question):
    # Step 1: LLM picks a tool
    prompt = f"""You are an assistant with access to these tools:

{build_tool_list()}

Reply with ONLY the tool name to answer this question. No explanation.

Question: {question}
Tool:"""

    tool_name = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    ).json()["response"].strip()

    # Step 2: call the tool
    tool_result = call_tool(tool_name)

    # Step 3: LLM answers using the tool result
    answer_prompt = f"""Answer the question using only the information below.

Information: {tool_result}

Question: {question}
Answer:"""

    answer = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": answer_prompt, "stream": False},
    ).json()["response"].strip()

    return tool_name, tool_result, answer

questions = [
    "How do I get my money back?",
    "How long does delivery take?",
]

for q in questions:
    tool_name, tool_result, answer = ask(q)
    print(f"Q:      {q}")
    print(f"Tool:   {tool_name}")
    print(f"Result: {tool_result}")
    print(f"Answer: {answer}")
    print()
```

```
Q:      How do I get my money back?
Tool:   get_refund_policy
Result: Refunds are processed within 5-7 business days. Contact support@company.com with your order number.
Answer: Refunds are processed within 5-7 business days. Contact support@company.com with your order number.

Q:      How long does delivery take?
Tool:   get_shipping_info
Result: Standard shipping 3-5 days. Express 1-2 days. Free shipping on orders over $50.
Answer: Delivery time depends on the shipping option chosen:

* Standard shipping: 3-5 days
* Express: 1-2 days
```

> **Key insight:** The three-step pattern is the foundation of every agent:
>
> ```
> Step 1: LLM picks a tool     ← reasoning
> Step 2: Call the tool        ← acting
> Step 3: LLM answers          ← synthesizing
> ```
>
> The first answer is a direct passthrough — the LLM had nothing to add.
> The second answer shows generation adding value — reformatting raw tool
> output into a structured response.
>
> In Block 2, the LLM will loop across multiple tools instead of
> stopping after one.

---
## Step 02 — The Agent Loop

> **NOTE:** Block 1 called exactly one tool per question and stopped.
> Block 2 lets the LLM decide how many tools to call before answering.
>
> ```
> Block 1: question → pick tool → call tool → answer   (always 1 tool)
> Block 2: question → pick tool → call tool → "enough?"
>                                    ├── No  → pick tool → call tool → ...
>                                    └── Yes → answer
> ```
>
> This is the ReAct loop: Reason → Act → Observe → Reason → ...

### The ANSWER stop signal

```
TOOLS list + ANSWER signal → LLM picks one per step
  picked == tool   → call it, add to history, loop again
  picked == ANSWER → generate final answer from history, stop
```

Already-called tools are removed from the list in code each step —
the LLM cannot pick what isn't shown to it. Enforcing constraints
in code is more reliable than asking the LLM to remember.

### Debugging the loop — what went wrong and how we fixed it

The loop did not work on the first attempt. This is the real lesson:

```
Problem:    LLM ignored the tool list, kept picking the same tool
Symptom:    Step 1-5 all picked 'get_shipping_info'
Diagnosis:  added prints to see called_tools and remaining_tools each step
Root cause: llama3 returned 'get_shipping_info' with quotes — the quoted
            version was added to called_tools but never matched the
            unquoted name in remaining_tools, so it was never removed
Fix 1:      strip quotes with .strip("'\"") after getting LLM response
Fix 2:      rewrite prompt to be unambiguous about valid choices
```

Debugging print that revealed the problem:
```
Step 1: picked 'get_shipping_info' | called=set()                 | remaining=['get_refund_policy', 'get_shipping_info', 'get_return_policy']
Step 2: picked 'get_shipping_info' | called={'get_shipping_info'} | remaining=['get_refund_policy', 'get_return_policy']
Step 3: picked 'get_shipping_info' | called={'get_shipping_info'} | remaining=['get_refund_policy', 'get_return_policy']
```

`remaining` was shrinking correctly — but the LLM kept picking the
removed tool anyway. The prompt was not explicit enough.

> **Key insight:** Prompt engineering and code-side enforcement both
> matter. Never rely solely on the LLM to respect constraints —
> remove options in code so it physically cannot pick them.

### src/agents/02_agent_loop.py

```python
"""
Block 02 — The Agent Loop
=========================
Teaches the ReAct loop: Reason -> Act -> Observe -> Reason -> ...

In Block 1 the LLM always called exactly one tool then answered.
Here the LLM decides at each step whether to call another tool
or stop and return a final answer.

The key addition: a STOP signal called ANSWER.
When the LLM picks ANSWER instead of a tool, the loop ends.

Flow:
    1. Show LLM the question + remaining tool descriptions
    2. LLM picks a tool OR picks ANSWER
    3. If tool  -> call it, add result to history, go to step 1
    4. If ANSWER -> ask LLM to produce final answer from history
"""

import requests

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
MAX_STEPS    = 5  # safety limit — prevent infinite loops


# ── Tools ─────────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_tool(name):
    """Look up a tool by name and call it. Returns None if not found."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["function"]()
    return None


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(question):
    """
    ReAct loop — runs until the LLM picks ANSWER or MAX_STEPS is reached.

    Already-called tools are removed from the list each step so the LLM
    cannot pick them again. Constraints enforced in code, not by prompt.

    Args:
        question: the user's question as a plain string

    Returns:
        tuple of (final_answer, steps_taken)
    """
    history      = []
    called_tools = set()

    for step in range(MAX_STEPS):
        # Remove already-called tools — LLM cannot pick what isn't listed
        remaining_tools = [t for t in TOOLS if t["name"] not in called_tools]

        # Force ANSWER if all tools exhausted
        if not remaining_tools:
            picked = "ANSWER"
        else:
            tool_lines = "\n".join(
                f"- {t['name']}: {t['description']}" for t in remaining_tools
            )
            tool_lines += "\n- ANSWER: Use this when you have enough information to answer the question."

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
            # llama3 sometimes wraps the name in quotes — strip them
            picked = raw.strip().strip("'\"")

        print(f"  Step {step + 1}: picked '{picked}'")

        # Stop — generate final answer from accumulated history
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

        # Call tool, record result, mark as used
        result = call_tool(picked)
        if result:
            history.append({"tool": picked, "result": result})
            called_tools.add(picked)

    return "Could not answer within step limit.", MAX_STEPS


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "How long does delivery take?",
        "Can I return a sale item and get a refund?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer, steps = run_agent(q)
        print(f"A: {answer}")
        print(f"Steps taken: {steps}")
```

```
Q: How long does delivery take?
  Step 1: picked 'get_shipping_info'
  Step 2: picked 'ANSWER'
A: Standard shipping takes 3-5 days, and express shipping takes 1-2 days.
Steps taken: 2

Q: Can I return a sale item and get a refund?
  Step 1: picked 'get_return_policy'
  Step 2: picked 'get_refund_policy'
  Step 3: picked 'get_shipping_info'
A: No, sale items are final and cannot be returned or refunded.
Steps taken: 4
```

> **Key insight:** Simple questions stop early (2 steps). Complex
> questions gather more before stopping (4 steps). The agent
> over-gathered on question 2 — shipping info was unnecessary —
> but still reached the correct answer.
>
> ```
> 1 tool call  → simple question, enough after one tool
> 3 tool calls → complex question, agent gathered more than needed
> MAX_STEPS    → safety net, loop can never run forever
> ```
>
> In Block 3 we add memory so the agent builds context across
> multiple questions, not just within a single run.

---
## Step 03 — Agent Memory

> **NOTE:** Blocks 1 and 2 had no memory. Every question started fresh —
> the agent forgot everything from previous questions.
>
> ```
> Block 2: Q1 → tools → answer    (forgets everything)
>          Q2 → tools → answer    (starts from scratch)
>
> Block 3: Q1 → tools → answer    (remembers)
>          Q2 → sees Q1 context → tools → better answer
> ```
>
> Built in 3 steps:
> ```
> Step 1: Show the problem — no memory means no context
> Step 2: Add conversation history — LLM sees all prior turns
> Step 3: Wire memory into the agent loop
> ```

### Two kinds of history

```
conversation_history → persists across questions (the full conversation)
step_history         → tool results within one question only (discarded after)

LLM sees both when picking tools and generating the final answer.
```

### Step 1 — The problem without memory

```
Q: How long does shipping take?
A: Standard shipping takes 3-5 days.

Q: What about express?
A: Express is a popular Node.js framework...   ← wrong, no context
```

"What about express?" got a response about Express.js — the LLM had
zero context from the previous question.

### Step 2 — Conversation history fixes it

```python
def ask_with_memory(question, history):
    # Add new question to history
    history.append({"role": "user", "content": question})

    # Format full conversation as a single prompt
    conversation = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history
    )
    prompt = f"{conversation}\nAssistant:"

    answer = ... # call Ollama

    # Store answer so next question can see it
    history.append({"role": "assistant", "content": answer})
    return answer
```

```
Q: How long does shipping take?
A: Standard shipping takes 3-5 days.

Q: What about express?
A: Express shipping typically takes 1-2 business days.  ← correct
```

> **Key insight:** The history list is passed by reference and modified
> in place. Every call appends two items — the question and the answer —
> so the next call sees the full conversation automatically.

### src/agents/03_agent_memory.py

```python
"""
Block 03 — Agent Memory
=======================
Teaches why memory matters and how to add it.

Without memory, every question starts fresh — the agent cannot
handle follow-up questions or build on previous answers.

With memory, the agent carries a conversation history that grows
with each question, giving the LLM full context at every step.

Two kinds of history:
    conversation_history → persists across questions (the full conversation)
    step_history         → tool results within one question, discarded after

Flow:
    No memory:   question → tools → answer → forget
    With memory: question → history + tools → answer → remember
"""

import requests

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
MAX_STEPS    = 5

# ... tools and TOOLS list same as Block 2 ...

def run_agent_with_memory(question, conversation_history):
    """
    ReAct loop with conversation memory.

    Args:
        question:             the current question
        conversation_history: list of prior turns, modified in place

    Returns:
        answer string
    """
    # Format prior conversation so the LLM has full context
    prior_context = ""
    if conversation_history:
        prior_context = "Previous conversation:\n" + "\n".join(
            f"  {t['role'].capitalize()}: {t['content']}"
            for t in conversation_history
        ) + "\n"

    step_history = []   # tool results for this question only
    called_tools = set()

    for step in range(MAX_STEPS):
        # ... same loop as Block 2, but pick_prompt includes prior_context ...

        if picked == "ANSWER":
            # Save Q&A to conversation history before returning
            conversation_history.append({"role": "user",      "content": question})
            conversation_history.append({"role": "assistant", "content": answer})
            return answer
```

```
AGENT WITH MEMORY:

Q: How long does shipping take?
  Step 1: picked 'get_shipping_info'
  Step 2: picked 'ANSWER'
A: Standard shipping takes 3-5 days. Express takes 1-2 days.

Q: What about express?
  Step 1: picked 'get_shipping_info'
  Step 2: picked 'ANSWER'
A: Express shipping takes 1-2 days.

Q: Can I return something I bought?
  Step 1: picked 'get_return_policy'
  Step 2: picked 'get_refund_policy'
  Step 3: picked 'ANSWER'
A: Yes, items can be returned within 30 days in original condition.

Q: And if it was on sale?
  Step 1: picked 'get_return_policy'
  Step 2: picked 'get_refund_policy'
  Step 3: picked 'get_shipping_info'
  Step 4: picked 'ANSWER'
A: Sale items are final and cannot be returned.

Conversation history: 8 turns
```

> **Key insight:** Follow-up questions ("What about express?",
> "And if it was on sale?") got correct answers because the LLM
> had the full prior conversation in its prompt.
>
> The follow-ups still called tools instead of answering purely
> from memory — llama3 doesn't always recognise when prior context
> is sufficient. A larger model skips the redundant tool calls.
> The answers are correct either way.
>
> ```
> conversation_history → 8 turns (2 per question: user + assistant)
> step_history         → reset each question, never persists
> ```
>
> In Block 4, RAG replaces the hardcoded tool functions — the agent
> will search your actual documents instead of returning fixed strings.

---
## Step 04 — RAG as a Tool

> **NOTE:** Blocks 1-3 used hardcoded tool functions returning fixed strings.
> Block 4 replaces those with a real RAG tool that searches ChromaDB.
>
> ```
> Block 3: get_shipping_info() → "Standard shipping 3-5 days."  (hardcoded)
> Block 4: search_docs(query)  → finds relevant chunks from your files
> ```
>
> The agent loop is unchanged. Only the tool changes.
>
> Built in 3 steps:
> ```
> Step 1: Load ChromaDB — verify 11 chunks still intact
> Step 2: Build search_docs tool — test it standalone before wiring in
> Step 3: Wire RAG tool into the agent loop
> ```

### Key difference from Block 2

In Block 2 the agent just picked a tool name. In Block 4 the tool
takes the question as input — the agent passes the question as the
search query to ChromaDB.

```
Block 2: agent picks "get_shipping_info" → calls get_shipping_info()
Block 4: agent picks "search_docs"       → calls search_docs(question)
```

### Step 2 output — RAG tool tested standalone

```
Query: How do I get a refund?
[score 0.513] (policy.txt) REFUND POLICY Refunds are processed within 5-7 business days...
[score 0.438] (policy.txt) To request a refund contact support@company.com...
[score 0.389] (policy.txt) sale and cannot be returned. To start a return...

Query: Can I return a sale item?
[score 0.698] (policy.txt) RETURN POLICY Items can be returned within 30 days...
[score 0.665] (faq.md) No. Sale items are final sale and cannot be returned...
[score 0.650] (policy.txt) sale and cannot be returned. To start a return...
```

> **Key insight:** Always test a tool standalone before wiring it into
> the agent. If the tool is broken the agent will fail in confusing ways.
> Confirm the tool works first, then connect it.

### src/agents/04_agent_rag.py

```python
"""
Block 04 — RAG as a Tool
========================
Replaces hardcoded tool functions with a real RAG tool that searches
your documents in ChromaDB.

The agent loop from Block 2 is unchanged.
The only difference is what the tool does when called.

Block 3: get_shipping_info() -> fixed string
Block 4: search_docs(query)  -> relevant chunks from ChromaDB

Flow:
    question -> agent loop -> search_docs tool -> ChromaDB
             -> retrieved chunks -> Ollama -> answer
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
CHROMA_PATH  = "src/rag/docs/chroma_db_rag"
COLLECTION   = "rag_docs"
MAX_STEPS    = 4

# ── Load embedder + ChromaDB once at startup ──────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

def search_docs(query, top_k=3):
    """
    Search ChromaDB for chunks relevant to the query.

    Same as RAG Block 6 retrieval, wrapped as an agent tool.

    Args:
        query:  search query (usually the user's question)
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
        chunks.append(f"[score {score}] ({meta['source']}) {doc}")
    return "\n".join(chunks)

# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name":        "search_docs",
        "description": "Search the company documents to find information about policies, shipping, returns, refunds, or any other company topic.",
        "function":    lambda: None,  # called with question as arg — see run_agent
    },
]

def run_agent(question):
    """
    ReAct loop with a single RAG tool.

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
                    f"  {h['result']}" for h in step_history
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
            chunks_text   = "\n".join(h["result"] for h in step_history)
            answer_prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{chunks_text}

Question: {question}
Answer:"""
            answer = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": answer_prompt, "stream": False},
            ).json()["response"].strip()
            return answer

        # Pass the question as the search query
        if picked == "search_docs":
            result = search_docs(question)
            step_history.append({"tool": picked, "result": result})
            called_tools.add(picked)

    return "Could not answer within step limit."


if __name__ == "__main__":
    questions = [
        "How do I get a refund?",
        "Can I return a sale item?",
        "What payment methods do you accept?",
        "How long does express shipping take?",
    ]

    print("\nAGENT WITH RAG TOOL:")
    print("─" * 60)

    for q in questions:
        print(f"\nQ: {q}")
        answer = run_agent(q)
        print(f"A: {answer}")
```

```
AGENT WITH RAG TOOL:

Q: How do I get a refund?
  Step 1: picked 'search_docs'
  Step 2: picked 'ANSWER'
A: To request a refund, contact support@company.com with your order number.

Q: Can I return a sale item?
  Step 1: picked 'search_docs'
  Step 2: picked 'ANSWER'
A: No, sale items are final sale and cannot be returned.

Q: What payment methods do you accept?
  Step 1: picked 'search_docs'
  Step 2: picked 'ANSWER'
A: We accept Visa, Mastercard, Amex, PayPal, and Apple Pay.

Q: How long does express shipping take?
  Step 1: picked 'search_docs'
  Step 2: picked 'ANSWER'
A: Express shipping takes 1-2 business days.
```

> **Key insight:** The agent loop never changed across Blocks 1-4.
> Only the tools changed — from fixed strings to real document search.
> This is the core design principle of agents:
>
> ```
> loop is generic   → same ReAct pattern every time
> tools are specific → swap them out for different capabilities
> ```
>
> Block 5 adds multiple tools so the agent can choose between
> RAG search, a calculator, and a date tool — picking the right
> one based on the question.

---
## Step 05 — Multiple Tools

> **NOTE:** Block 4 had one tool. Block 5 gives the agent three tools
> and lets it pick the right one based on the question type.
>
> ```
> Block 4: one tool   → search_docs (always the same choice)
> Block 5: three tools → agent decides which fits the question
>
>   "What is the refund policy?"  → search_docs  (needs documents)
>   "What is 15% tip on $84.99?"  → calculator   (needs math)
>   "What is today's date?"       → get_date     (needs system info)
> ```
>
> The agent loop is identical to Block 4. Only the tool registry changes.
>
> Built in 2 steps:
> ```
> Step 1: Define and test all three tools standalone
> Step 2: Wire into the agent loop — agent picks the right one
> ```

### Debugging — what went wrong and how we fixed it

The calculator tool failed on the first three attempts:

```
Attempt 1: LLM returned '0.15 × 84.99' — unicode × not valid Python
Attempt 2: LLM solved the math itself, returned '1.49825' as the expression
Attempt 3: Regex failed — re.error: nothing to repeat at position 40
Attempt 4: Simplified regex — still None, pct pattern didn't allow words between % and number
Fix:       Replace (?:tip on|of|off|on)? with [^0-9]* — matches any non-digit chars
```

Key lesson: don't ask the LLM to extract math expressions — do it in
code with regex. LLMs are unreliable at this task.

Final working pattern:
```python
# [^0-9]* matches anything between % and the next number
# covers: "15% tip on a 84.99", "20% off 150", "15% of 84.99"
pct_match = re.search(r'(\d+\.?\d*)\s*%[^0-9]*(\d+\.?\d*)', q)
```

### src/agents/05_agent_multi_tool.py

```python
"""
Block 05 — Multiple Tools
=========================
Gives the agent three tools and lets it pick the right one
based on the type of question asked.

Tools:
    search_docs  → searches ChromaDB for company policy questions
    calculator   → evaluates math expressions
    get_date     → returns today's date

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

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
CHROMA_PATH  = "src/rag/docs/chroma_db_rag"
COLLECTION   = "rag_docs"
MAX_STEPS    = 5

# ── Load embedder + ChromaDB ──────────────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)


# ── Tool functions ────────────────────────────────────────────────────────────

def search_docs(query, top_k=3):
    """Search ChromaDB for chunks relevant to the query."""
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
    """Safely evaluate a math expression using eval with no builtins."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"

def get_date():
    """Return today's date as a readable string."""
    return f"Today's date is {date.today().strftime('%A, %B %d, %Y')}."

def extract_expression(question):
    """
    Extract a math expression from a natural language question.
    Uses regex rather than asking the LLM — more reliable.

    [^0-9]* matches any non-digit characters between % and the number,
    so "15% tip on a 84.99", "20% off 150", "15% of 84.99" all match.
    """
    q = question.lower()
    q = q.replace("×", "*").replace("÷", "/").replace("−", "-")
    q = q.replace(" plus ", " + ").replace(" minus ", " - ")
    q = q.replace(" times ", " * ").replace(" divided by ", " / ")
    q = q.replace("$", "").replace(",", "")

    # Handle percentage questions with any words between % and number
    pct_match = re.search(r'(\d+\.?\d*)\s*%[^0-9]*(\d+\.?\d*)', q)
    if pct_match:
        pct   = float(pct_match.group(1)) / 100
        value = pct_match.group(2)
        return f"{pct} * {value}"

    # Fall back: extract digits and operators
    tokens = re.findall(r'[\d\.]+|[\+\-\*\/\(\)]', q)
    if len(tokens) >= 3:
        return " ".join(tokens)

    return None


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name":        "search_docs",
        "description": "Use this for questions about company policies, shipping, returns, refunds, payments, or anything requiring company information.",
        "function":    "rag",
    },
    {
        "name":        "calculator",
        "description": "Use this for any math calculation, percentages, totals, or arithmetic.",
        "function":    "math",
    },
    {
        "name":        "get_date",
        "description": "Use this when the user asks about today's date or the current day.",
        "function":    get_date,
    },
]


def call_tool(name, question):
    """Call the named tool with appropriate arguments."""
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


def run_agent(question):
    """
    ReAct loop with three tools.
    Agent picks the tool that matches the question type.
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
```

```
AGENT WITH MULTIPLE TOOLS:

Q: What is the refund policy?
  Step 1: picked 'search_docs'
  Step 2: picked 'calculator'
  Step 3: picked 'ANSWER'
A: Refunds are processed within 5-7 business days to the original payment method.
   Refund requests must be submitted within 60 days of purchase.
   Digital products are non-refundable once downloaded.

Q: What is 15% tip on a $84.99 order?
  Step 1: picked 'calculator'
  Step 2: picked 'search_docs'
  Step 3: picked 'get_date'
  Step 4: picked 'ANSWER'
A: The 15% tip on an $84.99 order is $12.75.

Q: What is today's date?
  Step 1: picked 'get_date'
  Step 2: picked 'ANSWER'
A: Sunday, March 01, 2026.

Q: Can I return a sale item?
  Step 1: picked 'search_docs'
  Step 2: picked 'ANSWER'
A: No, sale items are final sale and cannot be returned or exchanged.

Q: How much is 20% off a $150 item?
  Step 1: picked 'calculator'
  Step 2: picked 'search_docs'
  Step 3: picked 'get_date'
  Step 4: picked 'ANSWER'
A: 20% off $150 is $30 off, so you pay $120.
```

> **Key insight:** The agent loop never changed across all 5 blocks.
> Only the tools changed. This is the core design principle:
>
> ```
> loop is generic   → same ReAct pattern every time
> tools are specific → swap them in and out for any capability
> ```
>
> llama3 over-gathers on math questions — calls search_docs and
> get_date even after the calculator already has the answer.
> A larger model stops earlier. Answers are correct either way.
>
> The hardest part of Block 5 was not the agent loop — it was making
> the calculator tool reliable. Don't ask the LLM to extract math
> expressions. Do it in code with regex.

---

## Agents Complete

```
01 → tools           → define tools, LLM picks and calls them
02 → loop            → LLM loops until it has enough to answer
03 → memory          → agent remembers across questions
04 → RAG tool        → searches real documents in ChromaDB
05 → multiple tools  → agent picks the right tool for each question
```

The full agent pattern built across these blocks:

```python
while not done:
    pick tool from remaining tools   # LLM reasons
    call the tool                    # code acts
    add result to history            # observe
    if enough info: answer and stop  # LLM reasons again
```

Next: MCP Server — expose your RAG and agent tools as a standard
protocol that any MCP-compatible client can call.