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

    elif name == "Skill":
        skill_name = arguments["name"]
        skill_arguments = arguments.get("args", "")

        body = load_skill_body(skill_name)
        if body.startswith("Error:"):
            return body
        return substitute_skill_arguments(body, skill_arguments)

    elif name == "RunSkillScript":
        return run_skill_script(
            skill_name=arguments["skill_name"],
            script_path=arguments["script_path"],
            arguments=arguments.get("arguments", []),
        )

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

    prompt = "You have access to the following skills:\n"

    if skills:
        prompt += "You have access to the following skills:\n"

        for skill in skills:
            prompt += f"\n- {skill['name']}: {skill['description']}"

        prompt += (
            "\n\nIf a skill matches the user's request, call the Skill tool "
            "with its name and follow the instructions it returns."
        )

    return prompt


def load_skill_body(skill_name):
    skills_dir = Path(".claude/skills").resolve()
    skill_path = (skills_dir / skill_name / "SKILL.md").resolve()

    if not skill_path.is_relative_to(skills_dir):
        return f"Error: Invalid skill name: {skill_name}"

    if not skill_path.is_file():
        return f"Error: Skill '{skill_name}' not found."

    content = skill_path.read_text(encoding="utf-8")

    if not content.startswith("---"):
        return f"Error: Skill '{skill_name}' has invalid frontmatter."

    parts = content.split("---", 2)

    if len(parts) < 3:
        return f"Error: Skill '{skill_name}' has invalid frontmatter."

    # Return the raw body. Do not substitute arguments here.
    return parts[2].strip()


def substitute_skill_arguments(body, arguments):
    positional_arguments = arguments.split()

    # Replace indexed placeholders first, e.g. $ARGUMENTS[0].
    for index, argument in enumerate(positional_arguments):
        body = body.replace(f"$ARGUMENTS[{index}]", argument)

    # Replace the complete argument string.
    body = body.replace("$ARGUMENTS", arguments)

    # Replace shorthand positional placeholders, e.g. $0.
    for index, argument in enumerate(positional_arguments):
        body = body.replace(f"${index}", argument)

    return body


def resolve_prompt(prompt):
    if not prompt.startswith("/"):
        return None

    tokens = prompt.split()
    known_skill_names = {skill["name"] for skill in discover_skills()}

    skill_names = []
    index = 0

    # Identify consecutive skill names at the beginning.
    while index < len(tokens):
        token = tokens[index]

        if not token.startswith("/") or token == "/":
            break

        skill_name = token[1:]

        if skill_name not in known_skill_names:
            break

        skill_names.append(skill_name)
        index += 1

    if not skill_names:
        return None

    # Everything after the skill names is shared argument text.
    arguments = " ".join(tokens[index:])

    resolved_prompts = []

    for skill_name in skill_names:
        skill_body = load_skill_body(skill_name)

        if skill_body.startswith("Error:"):
            return [skill_body]

        skill_body = substitute_skill_arguments(
            skill_body,
            arguments,
        )

        resolved_prompts.append(skill_body)

    return resolved_prompts


def run_skill_script(skill_name, script_path, arguments=None):
    """
    Execute a script bundled with a skill.

    Scripts must be located inside the selected skill's directory.
    """
    skills_dir = Path(".claude/skills").resolve()
    skill_dir = (skills_dir / skill_name).resolve()

    # Validate the skill directory.
    if not skill_dir.is_relative_to(skills_dir) or not skill_dir.is_dir():
        return f"Error: Invalid or unknown skill: {skill_name}"

    # Resolve the requested script relative to the skill directory.
    script = (skill_dir / script_path).resolve()

    # Prevent path traversal and symlinks escaping the skill directory.
    if not script.is_relative_to(skill_dir):
        return "Error: Script must be inside the skill directory."

    if not script.is_file():
        return f"Error: Script not found: {script_path}"

    if not os.access(script, os.X_OK):
        return (
            f"Error: Script is not executable: {script_path}. "
            f"Make it executable with chmod +x."
        )

    if arguments is None:
        arguments = []

    if not isinstance(arguments, list) or not all(
        isinstance(arg, str) for arg in arguments
    ):
        return "Error: Script arguments must be a list of strings."

    try:
        result = subprocess.run(
            [str(script), *arguments],
            cwd=str(skill_dir),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        output = result.stdout

        if result.stderr:
            output += result.stderr

        if result.returncode != 0:
            output += f"\nScript exited with code {result.returncode}"

        return output or "(Script completed without output.)"

    except subprocess.TimeoutExpired:
        return "Error: Script execution timed out after 60 seconds."
    except OSError as exc:
        return f"Error executing script: {exc}"


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
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load a skill's instructions into the conversation",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the skill to use",
                        },
                        "args": {
                            "type": "string",
                            "description": "Optional arguments for the skill",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "RunSkillScript",
                "description": (
                    "Execute a script bundled with a skill. "
                    "Use this when the skill instructions require running one of "
                    "the scripts included in that skill's directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill containing the script.",
                        },
                        "script_path": {
                            "type": "string",
                            "description": (
                                "Path to the script relative to the skill directory, "
                                "for example scripts/deploy.sh."
                            ),
                        },
                        "arguments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional command-line arguments for the script.",
                        },
                    },
                    "required": ["skill_name", "script_path"],
                },
            },
        },
    ]

    skills = discover_skills()

    system_prompt = build_system_prompt(skills)

    resolved_prompts = resolve_prompt(args.p)

    messages = []

    if system_prompt:

        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    if resolved_prompts is not None:

        for resolved_prompt in resolved_prompts:

            messages.append(
                {
                    "role": "user",
                    "content": str(resolved_prompt),
                }
            )

    else:

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

        messages.append(message)

        if not message.tool_calls:
            if message.content:
                print(message.content)
            break

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
