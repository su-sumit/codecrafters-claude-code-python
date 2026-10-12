import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import yaml

from pathlib import Path
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

    elif name == "Write":
        file_path = arguments["file_path"]
        content = arguments["content"]

        Path(file_path).write_text(content)
        return f"Successfully wrote to {file_path}"

    elif name == "Bash":
        command = arguments["command"]

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout

        if result.stderr:
            output += result.stderr

        if result.returncode != 0:
            output += f"\nCommand exited with code {result.returncode}"

        return output

    raise ValueError(f"Unknown tool: {name}")


def discover_skills():
    skills_dir = Path(".claude/skills")
    skills = []

    if not skills_dir.is_dir():
        return skills

    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        content = skill_file.read_text()

        if not content.startswith("---"):
            continue

        parts = content.split("---", 2)

        if len(parts) < 3:
            continue

        try:
            metadata = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            continue

        if not isinstance(metadata, dict):
            continue

        name = metadata.get("name")
        description = metadata.get("description")

        if name and description:
            skills.append(
                {
                    "name": name,
                    "description": description,
                }
            )

    return skills


def build_system_prompt(skills):
    if not skills:
        return ""

    prompt = "You have access to the following skills:\n"

    for skill in skills:
        prompt += f"\n- {skill['name']}: {skill['description']}"

    return prompt


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
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file, creating it if necessary or overwriting it if it exists",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "The complete content to write to the file",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute",
                        }
                    },
                },
            },
        },
    ]

    skills = discover_skills()

    system_prompt = build_system_prompt(skills)

    messages = []

    if system_prompt:

        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": args.p,
        }
    )

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
