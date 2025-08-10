import os
import json
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load secrets from project-level .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "simple_price.json"
CHECK_PRICE_GRAMMAR_PATH = BASE_DIR / "check_price.lark"

DEFAULT_MODEL_NAME = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "minimal"
MODEL_NAME = os.getenv("OPENAI_MODEL", DEFAULT_MODEL_NAME)
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)


def main(user_prompt: str):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    client = OpenAI(api_key=OPENAI_API_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with open(CHECK_PRICE_GRAMMAR_PATH, "r") as f:
        check_price_grammar = f.read()

    system_prompt = """
You are a simple price checker. Given a product SKU, call the tool `checkPrice` to get a random price in USD cents, then summarize the price to the user in dollars (e.g., $12.34).
"""

    tools = [
        {
            "type": "custom",
            "name": "checkPrice",
            "description": "Return a random price in cents for the given SKU.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": check_price_grammar,
            },
        }
    ]

    def call_tool(name: str, args: dict):
        if name == "checkPrice":
            sku = args["sku"]
            price_cents = random.randint(500, 19999)
            return {"sku": sku, "priceCents": price_cents, "currency": "USD"}
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Initial request asking model to call the tool once
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info("Inference start: initial request (expect 1 custom tool call)")
    response = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        tools=tools,
        reasoning={"effort": REASONING_EFFORT},
    )

    # Detailed logging of what the model sent down in the first response
    logger.info("Initial response.output_text: %s", getattr(response, "output_text", None))
    for idx, item in enumerate(response.output):
        item_log = {
            "idx": idx,
            "type": getattr(item, "type", None),
            "name": getattr(item, "name", None),
            "call_id": getattr(item, "call_id", None),
            # For function_call (JSON tools)
            "arguments": getattr(item, "arguments", None),
            # For custom_tool_call (CFG tools)
            "input": getattr(item, "input", None),
            # For plain text chunks
            "content": getattr(item, "content", None),
        }
        try:
            logger.info("Initial response item: %s", json.dumps(item_log, indent=2))
        except Exception:
            logger.info("Initial response item (raw): %s", str(item_log))

    tool_call = None
    for item in response.output:
        if getattr(item, "type", None) == "custom_tool_call":
            tool_call = item
            break

    if tool_call is None:
        logger.info("Model did not request a tool call; writing snapshot and exiting")
        snapshot = {
            "prompt": user_prompt,
            "tool_called": False,
            "final_text": getattr(response, "output_text", None),
        }
        with open(OUTPUT_PATH, "w") as f:
            json.dump(snapshot, f, indent=2)
        return

    # Execute the single tool call (input is a JSON string per custom tool calls)
    args = json.loads(tool_call.input)
    result = call_tool(tool_call.name, args)

    # Append the tool result and ask the model once more for a concise summary
    messages += response.output
    messages.append(
        {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result),
        }
    )

    logger.info(
        "Inference start: final summary request (no further tool calls expected)"
    )
    response2 = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        tools=tools,
        reasoning={"effort": REASONING_EFFORT},
    )

    final_text = getattr(response2, "output_text", None)

    snapshot = {
        "prompt": user_prompt,
        "tool_call": {"name": tool_call.name, "args": args, "result": result},
        "final_text": final_text,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)

    if final_text:
        logger.info(f"Final output text: {final_text}")
    else:
        logger.info("Final output text unavailable on response")


if __name__ == "__main__":
    main("What's the price for SKU 'SKU-001' today?")
