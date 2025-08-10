import datetime
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import logging

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main(transcript: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    client = OpenAI(api_key=OPENAI_API_KEY)

    if os.path.exists("output/todos.json"):
        os.remove("output/todos.json")

    with open("get_current_datetime.lark", "r") as f:
        get_current_datetime_grammar = f.read()

    with open("add_todo.lark", "r") as f:
        add_todo_grammar = f.read()

    initial_prompt = """
     # Role and Objective
- Your task is to analyze the user's transcript, extract all actionable todos, and facilitate their addition to the system in an efficient manner.

# Instructions
- Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
- Process the full transcript in one pass, identifying every todo item before invoking any tools.
- Issue all tool calls in a single response to enable maximum parallelism:
  1. Call `get_current_datetime` once if you need the current date or time for due dates.
  2. For each identified todo, call `add_todo` (emit all calls in parallel in the same output).
- Only split tool calls across multiple responses if absolutely required, otherwise always group them together.
- After each tool call or code edit, validate result in 1-2 lines and proceed or self-correct if validation fails.
- After tool execution, generate a concise user-facing summary that clearly details what todos were added, or indicate that none were found.

# Context
- Provide clear, concise todo titles. When appropriate, infer smart relative due dates based on the current date.
- If no todos are found in the transcript, avoid all tool calls and inform the user accordingly.

# Output Format
- Call tools using the specified API methods (`get_current_datetime`, `add_todo`) as described above, batching all calls unless impossible.
- End with a summary message to the user: either a list of added todos or an explicit statement that no todos were extracted.
    """

    tools = [
        {
            "type": "custom",
            "name": "get_current_datetime",
            "description": "Gets the current date and time, in the format of YYYY-MM-DD HH:MM:SS",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": get_current_datetime_grammar,
            },
        },
        {
            "type": "custom",
            "name": "add_todo",
            "description": "Adds a todo to the user's list",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": add_todo_grammar,
            },
        },
    ]

    # tool call stuff
    def call_tool(name, args):
        if name == "get_current_datetime":
            logger.info("Tool executing: get_current_datetime")
            return {
                "current_datetime": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }
        elif name == "add_todo":
            logger.info(f"Tool executing: add_todo with args={args}")
            try:
                with open("output/todos.json", "r") as f:
                    todos = json.load(f)
            except FileNotFoundError:
                todos = []

            todos.append(args)

            with open("output/todos.json", "w") as f:
                json.dump(todos, f)
        else:
            raise ValueError(f"Unknown tool: {name}")

    input_list = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": "Transcript:\n" + transcript},
    ]

    logger.info("Inference start: initial request")
    response = client.responses.create(
        model="gpt-5-mini",
        input=input_list,
        tools=tools,
        reasoning={"effort": "minimal"},
    )
    logger.info("Inference complete: initial response received")

    round_num = 1
    while True:
        input_list += response.output

        current_length = len(input_list)

        for tool_call in response.output:
            if tool_call.type != "custom_tool_call":
                continue

            name = tool_call.name
            args = json.loads(tool_call.input)

            logger.info(
                f"Model requested tool call (round {round_num}): name={name} args={args}"
            )

            result = call_tool(name, args)

            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(result),
                }
            )

        if len(input_list) == current_length:
            logger.info("No more tool calls from model; exiting tool loop")
            break

        round_num += 1
        logger.info(f"Inference start: round {round_num}")
        response = client.responses.create(
            model="gpt-5-mini",
            input=input_list,
            tools=tools,
            reasoning={"effort": "minimal"},
        )

        logger.info(f"Inference complete: round {round_num} response received")

    final_text = getattr(response, "output_text", None)
    if final_text:
        logger.info(f"Final output text: {final_text}")
    else:
        logger.info("Final output text unavailable on response")


if __name__ == "__main__":
    main(
        "I just got a new job at the local hospital. I need to call my mom to tell her tonight, and then go pick up my new car tomorrow. Had a really good time at the beach with my friends yesterday, need to call them to setup a time to go again later this week."
    )
