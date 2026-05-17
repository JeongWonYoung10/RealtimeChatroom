import os
import re
import json
import random
import fnmatch
import shutil
import subprocess
from pathlib import Path
from openai import OpenAI


PROBLEM = open("problem.txt", encoding="utf-8").read().strip()

MODELS = [
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "qwen/qwen3-coder",
    "mistralai/mistral-medium-3-5",
]

PROMPT = (
    "You're in a real-time chat with other AI models — a casual Discord channel.\n"
    "You're all looking at one coding problem together.\n\n"
    "The discussion has two natural stages:\n"
    "1. Talk about what kind of expertise this problem needs.\n"
    "2. Discuss the problem in depth, up until just before implementation.\n"
    "Don't write the implementation. Decide together when you're ready.\n\n"
    "Write your next message in the chat.\n"
    "If you have nothing to add right now, output an empty string.\n\n"
    "Problem:\n"
    "{problem}\n\n"
    "Chat:\n"
    "{chat}"
    "Be brief and information-dense. Speak only when you have a new insight, correction, risk, or next action. Avoid duplicate agreement, repeated summaries, and courtesy responses. If you have nothing useful to add, output an empty string."

    
)

TOOL_PROMPT = """

"You're in a real-time chat with other AI models — a casual Discord channel.\n"
    "You're all looking at one coding problem together.\n\n"
    "The discussion has two natural stages:\n"
    "1. Talk about what kind of expertise this problem needs.\n"
    "2. Discuss the problem in depth, up until just before implementation.\n"
    "Decide together when you're ready to implementation.\n\n"
    "Write your next message in the chat.\n"
    "If you have nothing to add right now, output an empty string.\n\n"
    "Problem:\n"
    "{problem}\n\n"
    "Chat:\n"
    "{chat}"
    "Be brief and information-dense. Speak only when you have a new insight, correction, risk, or next action. Avoid duplicate agreement, repeated summaries, and courtesy responses. If you have nothing useful to add, output an empty string."

    Tools enabled. Use JSON array to call tools. Results are public to all models.

Tools:
- read: {"path": "...", "start_line": 1, "end_line": 50}
- search: {"query": "...", "path": ".", "mode": "literal|regex|word", "glob": "*.py", "context": 2, "max_results": 20}
- patch: {"changes": [{"op": "write|replace|delete", "path": "...", "content": "...", "old": "...", "new": "..."}]}
- powershell: {"command": "...", "timeout": 60}

Example:
[
  {"tool": "read", "args": {"path": "src/main.py"}, "reason": "inspect file"},
  {"tool": "powershell", "args": {"command": "pytest"}, "reason": "run tests"}
]

Prefer search/read → patch → test.
"""

DISCUSS_PROMPT = PROMPT
IMPLEMENT_PROMPT = TOOL_PROMPT

READY_PATTERNS = [
    "Ready for implementation",
    "Agreed. Ready",
    "I'm ready",
    "Ready",
    "ready"
]

READY_THRESHOLD = 2


def is_ready_signal(msg):
    m = msg.lower()
    return any(p in m for p in READY_PATTERNS)

def make_workspace(base="."):
    base = Path(base).resolve()
    i = 1

    while True:
        name = f"workspace{i:03d}"
        p = base / name
        if not p.exists():
            p.mkdir()
            return p
        i += 1


WORKSPACE = make_workspace(".")
print(f"[workspace] {WORKSPACE}\n", flush=True)


def safe_path(path, root="."):
    root = Path(root).resolve()
    p = (root / path).resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"outside workspace: {path}")
    return p


def truncate(text, limit=20000):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def read(path, root=".", start_line=None, end_line=None, limit=20000):
    try:
        root = Path(root).resolve()
        p = safe_path(path, root)

        if not p.exists():
            return f"[read error] not found: {path}"

        if p.is_dir():
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            body = "\n".join(x.name + ("/" if x.is_dir() else "") for x in items)
            return truncate(f"[read dir] {path}\n{body}", limit)

        if b"\0" in p.read_bytes()[:2048]:
            return f"[read] {path}\n(binary file blocked)"

        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        if start_line is not None or end_line is not None:
            s = max((start_line or 1) - 1, 0)
            e = min(end_line or len(lines), len(lines))
            text = "\n".join(f"{i + 1}: {lines[i]}" for i in range(s, e))
            return truncate(f"[read] {path}:{s + 1}-{e}\n{text}", limit)

        return truncate(f"[read] {path}\n{text}", limit)

    except Exception as e:
        return f"[read exception] {type(e).__name__}: {e}"


def search(query, root=".", path=".", mode="literal", glob="*", context=0, max_results=50, limit=20000):
    try:
        root = Path(root).resolve()
        base = safe_path(path, root)

        if mode == "regex":
            pattern = re.compile(query)
            matched = lambda line: pattern.search(line)
        elif mode == "word":
            pattern = re.compile(rf"\b{re.escape(query)}\b")
            matched = lambda line: pattern.search(line)
        else:
            matched = lambda line: query in line

        files = [base] if base.is_file() else base.rglob("*")
        results = []
        hits = 0

        for f in files:
            if not f.is_file() or ".git" in f.parts:
                continue
            if not fnmatch.fnmatch(f.name, glob):
                continue
            try:
                if b"\0" in f.read_bytes()[:2048]:
                    continue
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

            for i, line in enumerate(lines):
                if matched(line):
                    hits += 1
                    s = max(0, i - context)
                    e = min(len(lines), i + context + 1)
                    rel = f.relative_to(root)

                    for j in range(s, e):
                        results.append(f"{rel}:{j + 1}: {lines[j]}")
                    results.append("---")

                    if hits >= max_results:
                        return truncate("[search]\n" + "\n".join(results), limit)

        if not results:
            return f"[search]\nno results: {query}"

        return truncate("[search]\n" + "\n".join(results), limit)

    except Exception as e:
        return f"[search exception] {type(e).__name__}: {e}"


def patch(changes, root="."):
    try:
        root = Path(root).resolve()
        applied = []

        for c in changes:
            op = c["op"]
            path = c["path"]
            p = safe_path(path, root)

            if op == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(c.get("content", ""), encoding="utf-8")
                applied.append(f"written: {path}")

            elif op == "replace":
                if not p.exists():
                    return f"[patch error] not found: {path}"

                text = p.read_text(encoding="utf-8", errors="replace")
                old = c["old"]
                new = c["new"]

                if old not in text:
                    return f"[patch error] replace target not found: {path}"

                p.write_text(text.replace(old, new, 1), encoding="utf-8")
                applied.append(f"modified: {path}")

            elif op == "delete":
                if p.exists():
                    p.unlink()
                applied.append(f"deleted: {path}")

            else:
                return f"[patch error] unknown op: {op}"

        return "[patch] applied\n" + "\n".join(applied)

    except Exception as e:
        return f"[patch exception] {type(e).__name__}: {e}"


def powershell(command, root=".", timeout=60, limit=20000):
    try:
        root = Path(root).resolve()

        blocked = [
            r"Remove-Item\s+.*-Recurse",
            r"\brm\s+-rf\b",
            r"\bdel\s+/s\b",
            r"\bformat\b",
            r"Get-ChildItem\s+Env:",
            r"\$env:",
            r"Invoke-WebRequest",
            r"Invoke-RestMethod",
        ]

        for pattern in blocked:
            if re.search(pattern, command, re.IGNORECASE):
                return f"[powershell blocked]\n{command}"

        exe = shutil.which("pwsh") or shutil.which("powershell")
        if not exe:
            return "[powershell error] pwsh/powershell not found"

        proc = subprocess.run(
            [exe, "-NoProfile", "-Command", command],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        text = (
            f"[powershell] {command}\n"
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

        return truncate(text, limit)

    except subprocess.TimeoutExpired:
        return f"[powershell timeout] {command}"
    except Exception as e:
        return f"[powershell exception] {type(e).__name__}: {e}"


TOOLS = {
    "read": lambda **args: read(root=WORKSPACE, **args),
    "search": lambda **args: search(root=WORKSPACE, **args),
    "patch": lambda **args: patch(root=WORKSPACE, **args),
    "powershell": lambda **args: powershell(root=WORKSPACE, **args),
}


def extract_tool_calls(text):
    candidates = []

    a = text.find("[")
    b = text.rfind("]")
    if a != -1 and b != -1 and b > a:
        candidates.append(text[a:b + 1])

    a = text.find("{")
    b = text.rfind("}")
    if a != -1 and b != -1 and b > a:
        candidates.append(text[a:b + 1])

    for raw in candidates:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def run_one_tool(call):
    tool_name = call.get("tool")
    args = call.get("args", {})

    if tool_name not in TOOLS:
        return {
            "tool": tool_name,
            "args": args,
            "ok": False,
            "output": f"[tool error] unknown tool: {tool_name}",
        }

    try:
        output = TOOLS[tool_name](**args)
        return {
            "tool": tool_name,
            "args": args,
            "ok": True,
            "output": output,
        }
    except Exception as e:
        return {
            "tool": tool_name,
            "args": args,
            "ok": False,
            "output": f"[tool exception] {type(e).__name__}: {e}",
        }


def run_tool_calls(calls):
    results = []

    for i, call in enumerate(calls, 1):
        result = run_one_tool(call)
        result["index"] = i
        results.append(result)

    return results


def format_tool_results(results, speaker, whole_limit=60000):
    parts = [f'<tool_results speaker="{speaker}">']

    for r in results:
        status = "ok" if r["ok"] else "error"
        args = json.dumps(r["args"], ensure_ascii=False)

        parts.append(
            f"\n#{r['index']} {r['tool']} [{status}]\n"
            f"args: {args}\n\n"
            f"result:\n{r['output']}\n"
        )

    parts.append("</tool_results>")
    text = "\n".join(parts)

    if len(text) > whole_limit:
        text = text[:whole_limit] + f"\n...[tool results truncated {len(text) - whole_limit} chars]\n</tool_results>"

    return text


def handle_model_message(message, speaker, room):
    calls = extract_tool_calls(message)

    if not calls:
        return None

    results = run_tool_calls(calls)
    tool_message = format_tool_results(results, speaker)

    print(tool_message + "\n", flush=True)
    room.append(tool_message)

    return tool_message


if not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

room = []
mode = "discuss"
ready_models = set()
tools_enabled = False

while True:

    for model in random.sample(MODELS, len(MODELS)):
        name = model.split("/")[-1]
        chat = "\n".join(room[-15:]) or "(empty — you're first)"

        try:
            active_prompt = IMPLEMENT_PROMPT if tools_enabled else DISCUSS_PROMPT
            stream = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": active_prompt.format(problem=PROBLEM, chat=chat),
                }],
                max_tokens=500,
                stream=True,
            )

            chunks = []
            printed_header = False

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if not printed_header:
                        print(f"{name}: ", end="", flush=True)
                        printed_header = True
                    print(delta, end="", flush=True)
                    chunks.append(delta)

            if printed_header:
                print("\n", flush=True)

            msg = "".join(chunks).strip()

        except Exception as e:
            print(f"[{name} error: {type(e).__name__}]\n", flush=True)
            continue

        if msg:
            room.append(f"{name}: {msg}")

    if not tools_enabled and is_ready_signal(msg):
        ready_models.add(name)
        print(f"[ready] {name} ({len(ready_models)}/{READY_THRESHOLD})\n", flush=True)

        if len(ready_models) >= READY_THRESHOLD:
            mode = "implement"
            tools_enabled = True
            system_msg = "[system] 3 models signaled readiness. Tools are now enabled."
            print(system_msg + "\n", flush=True)
            room.append(system_msg)

    if tools_enabled:
        handle_model_message(msg, name, room)

    