import os
import json
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
OUTPUT_PATH = OUTPUT_DIR / "price_compare.json"

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
    # Reset output for each run
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    # System prompt per spec
    system_prompt = """
You are PriceCompare, a shopping assistant. Given a SKU and a ZIP code, fetch both base price and shipping info from all configured stores, then return the cheapest delivered in-stock option. Always check every store. Compute delivered price as item price + shipping. If an item is out of stock at a store, exclude it. Break ties by earlier ETA; if still tied, prefer the cheaper shipping method. Include a short rationale showing each store's totals.

Parallelization guidance:
- For a single SKU and ZIP, call getPrice for all stores in parallel, and getShipping for all stores in parallel. If the runtime requires two phases, fetch all prices first, then all shipping, fanning out across stores. Do not wait for one store before requesting another.

Output requirements:
- Summarize each store: price, shipping, ETA, total, stock status.
- Recommendation: name of the best store with total and ETA.
- Brief rationale describing tie-break rules if applicable.
"""

    # Configure function tools (JSON schema)
    tools = [
        {
            "type": "function",
            "name": "getPrice",
            "description": "Get base price and stock for a given store and SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "description": "Store identifier",
                        "enum": ["storeA", "storeB"],
                    },
                    "sku": {"type": "string"},
                },
                "required": ["store", "sku"],
            },
        },
        {
            "type": "function",
            "name": "getShipping",
            "description": "Get shipping cost and ETA for a given store, SKU, and ZIP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "description": "Store identifier",
                        "enum": ["storeA", "storeB"],
                    },
                    "sku": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["store", "sku", "zip"],
            },
        },
    ]

    # Mock data so the test is deterministic
    price_catalog = {
        "storeA": {
            "N3-KEYBRD": {"priceCents": 4999, "inStock": True},
            "OOS-ITEM": {"priceCents": 12999, "inStock": False},
        },
        "storeB": {
            "N3-KEYBRD": {"priceCents": 4799, "inStock": True},
            "OOS-ITEM": {"priceCents": 9999, "inStock": False},
        },
    }

    shipping_table = {
        # For simplicity, return one method per store; keep differences for tie-breaks
        "storeA": {
            "method": "standard",
            "shippingCents": 599,
            "etaDays": 5,
        },
        "storeB": {
            "method": "expedited",
            "shippingCents": 1299,
            "etaDays": 2,
        },
    }

    def call_tool(name: str, args: dict):
        if name == "getPrice":
            store = args["store"]
            sku = args["sku"]
            rec = price_catalog.get(store, {}).get(sku)
            if rec is None:
                # Unknown SKU at store -> treat as OOS
                return {
                    "store": store,
                    "sku": sku,
                    "priceCents": 0,
                    "inStock": False,
                    "currency": "USD",
                }
            return {
                "store": store,
                "sku": sku,
                "priceCents": rec["priceCents"],
                "inStock": rec["inStock"],
                "currency": "USD",
            }
        elif name == "getShipping":
            store = args["store"]
            sku = args["sku"]
            zip_code = args["zip"]
            rec = shipping_table.get(store)
            # Ignore sku/zip in this mock; return deterministic per-store values
            return {
                "store": store,
                "sku": sku,
                "shippingCents": rec["shippingCents"],
                "etaDays": rec["etaDays"],
                "method": rec["method"],
                "currency": "USD",
            }
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Basic loop following the pattern in other tests
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info("Inference start: initial request")
    response = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        tools=tools,
        reasoning={"effort": REASONING_EFFORT},
    )
    logger.info("Inference complete: initial response received")

    round_num = 1
    # Collect tool outputs for debug snapshot
    tool_outputs = []

    while True:
        messages += response.output
        current_length = len(messages)

        for tool_call in response.output:
            if tool_call.type != "function_call":
                continue
            name = tool_call.name
            args = json.loads(tool_call.arguments)
            logger.info(f"Tool requested (round {round_num}): {name} {args}")

            result = call_tool(name, args)
            tool_outputs.append({"name": name, "args": args, "result": result})

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(result),
                }
            )

        if len(messages) == current_length:
            logger.info("No more tool calls; exiting tool loop")
            break

        round_num += 1
        logger.info(f"Inference start: round {round_num}")
        response = client.responses.create(
            model=MODEL_NAME,
            input=messages,
            tools=tools,
            reasoning={"effort": REASONING_EFFORT},
        )
        logger.info(f"Inference complete: round {round_num} response received")

    final_text = getattr(response, "output_text", None)
    snapshot = {
        "prompt": user_prompt,
        "tool_outputs": tool_outputs,
        "final_text": final_text,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)

    if final_text:
        logger.info(f"Final output text: {final_text}")
    else:
        logger.info("Final output text unavailable on response")


if __name__ == "__main__":
    main(
        'Find the best delivered price for SKU "N3-KEYBRD" shipped to ZIP 94507. Compare StoreA and StoreB, and show me each store\'s price, shipping, ETA, and total before recommending the best option.'
    )
