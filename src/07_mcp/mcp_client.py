"""
MCP Client
==========
A Python client that talks to any MCP server.

Mirrors exactly what Claude Desktop and Cursor do when they
connect to an MCP server:
    1. list_tools()       -> discover what's available
    2. pick_tool()        -> LLM decides which tool fits
    3. call_tool()        -> execute and get result
    4. answer()           -> LLM answers using the tool result

This client is reusable across all MCP blocks — import it,
point it at a server URL, and it works with any tool registry.
"""

import re
import requests

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3"


class MCPClient:
    """
    A minimal MCP client.

    Connects to an MCP server, discovers its tools, and uses an LLM
    to pick and call the right tool for a given question.

    Args:
        server_url: base URL of the MCP server e.g. "http://localhost:8001"
    """

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.tools      = []   # populated by list_tools()

    def list_tools(self):
        """
        Discover all tools available on the server.
        Populates self.tools and returns the list.
        """
        response   = requests.post(f"{self.server_url}/tools/list")
        self.tools = response.json()["tools"]
        return self.tools

    def call_tool(self, name: str, arguments: dict = {}):
        """
        Call a tool by name with arguments.

        Args:
            name:      tool name as returned by list_tools()
            arguments: dict of argument name -> value

        Returns:
            the tool result as a string
        """
        response = requests.post(
            f"{self.server_url}/tools/call",
            json={"name": name, "arguments": arguments},
        )
        data = response.json()
        if "content" not in data:
            return f"Unexpected server response: {data}"
        return data["content"][0]["text"]

    def pick_tool(self, question: str):
        """
        Ask the LLM which tool to call for this question.

        Formats the tool list into a prompt and asks the LLM to
        reply with exactly one tool name.

        Args:
            question: the user's question

        Returns:
            tool name string as picked by the LLM
        """
        tool_lines = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self.tools
        )

        prompt = f"""You have access to these tools:

{tool_lines}

Reply with ONLY the tool name that best answers this question.
No explanation. One word only.

Question: {question}
Tool:"""

        raw = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        ).json()["response"]

        return raw.strip().strip("'\"")

    def answer(self, question: str, tool_result: str):
        """
        Ask the LLM to answer the question using the tool result.

        Args:
            question:    the user's original question
            tool_result: the string returned by the tool

        Returns:
            the LLM's final answer as a string
        """
        prompt = f"""Answer the question using only the information below.
If the answer is not in the information, say "I don't know".

Information: {tool_result}

Question: {question}
Answer:"""

        return requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        ).json()["response"].strip()

    def _extract_expression(self, question: str):
        """
        Extract a math expression from a natural language question.
        Uses regex — more reliable than asking the LLM to do it.

        Handles:
            "20% of $150"        -> "0.2 * 150"
            "15% tip on $84.99"  -> "0.15 * 84.99"
            "42 * 3"             -> "42 * 3"

        Args:
            question: natural language math question

        Returns:
            a Python-evaluable expression string
        """
        q = question.lower().replace("$", "").replace(",", "")
        q = q.replace("×", "*").replace("÷", "/")

        # Handle percentage questions — allow any words between % and number
        pct_match = re.search(r'(\d+\.?\d*)\s*%[^0-9]*(\d+\.?\d*)', q)
        if pct_match:
            pct   = float(pct_match.group(1)) / 100
            value = pct_match.group(2)
            return f"{pct} * {value}"

        # Fall back — extract digits and operators
        tokens = re.findall(r'[\d\.]+|[\+\-\*\/\(\)]', q)
        if len(tokens) >= 3:
            return " ".join(tokens)

        # Last resort — pass the full question and let calculate() error gracefully
        return question

    def ask(self, question: str):
        """
        Full MCP flow for one question:
            discover -> pick tool -> build arguments -> call tool -> answer

        For tools that require arguments (e.g. calculate), extracts
        the argument value from the question before calling.

        Args:
            question: the user's question

        Returns:
            tuple of (tool_name, tool_result, answer)
        """
        # Refresh tool list so server changes are always picked up
        self.list_tools()

        tool_name = self.pick_tool(question)

        # Build arguments based on which tool was picked.
        # Tools with no required inputSchema fields get empty args.
        # calculate needs the math expression extracted from the question.
        arguments = {}
        if tool_name == "calculate":
            arguments = {"expression": self._extract_expression(question)}

        tool_result = self.call_tool(tool_name, arguments)
        answer      = self.answer(question, tool_result)

        return tool_name, tool_result, answer