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

DEFAULT_MODEL_NAME = "gpt-5"
DEFAULT_REASONING_EFFORT = "high"
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

    logger.info(
        "\n=== Inference start: initial request (expect 1 custom tool call) [STREAMING] ===\n"
    )

    # Use the streaming API context manager so we can retrieve the final assembled response
    tool_input_buffers: dict[str, str] = {}
    with client.responses.stream(
        model=MODEL_NAME,
        input=messages,
        tools=tools,
        reasoning={"effort": REASONING_EFFORT},
    ) as stream:
        for event in stream:
            # Prefer a compact, human-friendly log for each event
            payload = None
            try:
                payload = event.model_dump() if hasattr(event, "model_dump") else None
            except Exception:
                payload = None

            etype = payload.get("type") if isinstance(payload, dict) else None

            if etype == "response.output_item.added":
                item = payload.get("item", {})
                logger.info(
                    "\n[output_item.added] type=%s name=%s id=%s status=%s\n",
                    item.get("type"),
                    item.get("name"),
                    item.get("id"),
                    item.get("status"),
                )
            elif etype == "response.output_item.done":
                item = payload.get("item", {})
                logger.info(
                    "[output_item.done]   type=%s name=%s id=%s status=%s\n",
                    item.get("type"),
                    item.get("name"),
                    item.get("id"),
                    item.get("status"),
                )
            elif etype == "response.custom_tool_call_input.delta":
                item_id = payload.get("item_id")
                delta = payload.get("delta", "")
                if item_id:
                    tool_input_buffers[item_id] = (
                        tool_input_buffers.get(item_id, "") + delta
                    )
            elif etype == "response.custom_tool_call_input.done":
                item_id = payload.get("item_id")
                full_input = payload.get("input") or tool_input_buffers.get(item_id, "")
                logger.info(
                    "[tool_call.input]   %s\n",
                    full_input,
                )
            elif etype == "response.in_progress":
                logger.info("[response] in_progress")
            elif etype == "response.completed":
                resp = payload.get("response", {})
                usage = resp.get("usage") or {}
                logger.info(
                    "\n=== Stream completed ===\nstatus=completed, output_items=%d, tokens(input=%s, output=%s, total=%s)\n",
                    len(resp.get("output", []) or []),
                    (usage.get("input_tokens") if isinstance(usage, dict) else None),
                    (usage.get("output_tokens") if isinstance(usage, dict) else None),
                    (usage.get("total_tokens") if isinstance(usage, dict) else None),
                )
            else:
                # Keep other events minimal to avoid noise
                if etype:
                    logger.info("[event] %s", etype)

        # After the stream ends, obtain the assembled final response
        response = stream.get_final_response()

    # Concise logging of the assembled first response
    logger.info(
        "\nInitial response.output_text: %s\n",
        getattr(response, "output_text", None),
    )
    for idx, item in enumerate(response.output):
        logger.info(
            "[#%d] type=%s name=%s call_id=%s\n",
            idx,
            getattr(item, "type", None),
            getattr(item, "name", None),
            getattr(item, "call_id", None),
        )

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
