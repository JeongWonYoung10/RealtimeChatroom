"""AI chatroom: four OpenRouter models discussing one coding problem.
Run, watch, Ctrl+C when done.
"""
import os
import random
from openai import OpenAI

PROBLEM = open("problem.txt").read().strip()

MODELS = [
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "qwen/qwen3-coder",
    "mistralai/mistral-medium-3-5",
]

PROMPT = """You're in a real-time chat with other AI models — a casual Discord channel.
You're all looking at one coding problem together.

The discussion has two natural stages:
1. Talk about what kind of expertise this problem needs.
2. Discuss the problem in depth, up until just before implementation.
Don't write the implementation. Decide together when you're ready.

Write your next message in the chat.
If you have nothing to add right now, output an empty string.

Problem:
{problem}

Chat:
{chat}"""

if not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
room = []

while True:
    for model in random.sample(MODELS, len(MODELS)):
        name = model.split("/")[-1]
        chat = "\n".join(room[-50:]) or "(empty — you're first)"
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": PROMPT.format(problem=PROBLEM, chat=chat)}],
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