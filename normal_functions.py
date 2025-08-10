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

    if os.path.exists("todos.json"):
        os.remove("todos.json")

    # Using traditional function tools; CFG grammars are no longer needed

    initial_prompt = """
    You are a helpful internal system whose job is to read the user's transcript and extract todos to add.

    Requirements:
    - Parse the entire transcript first, identifying ALL todos before calling tools.
    - When adding todos, emit all tool calls together in a single response so they can be called in parallel:
      1) If needed, call get_current_datetime at most once to get the current date/time.
      2) Then, call add_todo for each todo you found
    - Do not split todo additions across multiple responses unless absolutely necessary. CALL THEM IN PARALLEL ALWAYS UNLESS YOU ABSOLUTELY CAN'T.
    - After tools are executed, return a final user-facing summary message of what you added (or that there were no todos).

    Notes:
    - Use clear titles. Infer reasonable due dates relative to the current date when appropriate.
    - If there are no todos, do not call any tools; just respond that there are none to add.
    """

    tools = [
        {
            "type": "function",
            "name": "get_current_datetime",
            "description": "Gets the current date and time, in the format of YYYY-MM-DD HH:MM:SS",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "add_todo",
            "description": "Adds a todo to the user's list",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the todo",
                    },
                    "due": {
                        "type": ["string", "null"],
                        "description": "Due date in YYYY-MM-DD or null",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["title", "due", "priority"],
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
                with open("todos.json", "r") as f:
                    todos = json.load(f)
            except FileNotFoundError:
                todos = []

            todos.append(args)

            with open("todos.json", "w") as f:
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
            if tool_call.type != "function_call":
                continue

            name = tool_call.name
            args = json.loads(tool_call.arguments)

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
