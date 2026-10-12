import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def execute_tool(tool_call):
    function = tool_call.function
    name = function.name
    arguments = json.loads(function.arguments)

    if name == "Read":
        file_path = arguments["file_path"]
        return Path(file_path).read_text()

    raise ValueError(f"Unknown tool: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
            max_tokens=5000,
        )

        message = chat.choices[0].message

        # Add the assistant's response to the conversation history.
        messages.append(message)

        # If there are no tool calls, the model has finished.
        if not message.tool_calls:
            if message.content:
                print(message.content)
            break

        # Execute each requested tool and send its result back.
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


if __name__ == "__main__":
    main()
