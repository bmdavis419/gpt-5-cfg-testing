import os
import json
from pathlib import Path
from datetime import datetime, timedelta, time
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load secrets from project-level .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "email_triage.json"

DEFAULT_MODEL_NAME = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "minimal"
MODEL_NAME = os.getenv("OPENAI_MODEL", DEFAULT_MODEL_NAME)
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)


def next_7_day_range_iso(now: datetime) -> tuple[str, str]:
    start = now
    end = now + timedelta(days=7)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")


def business_hours_slots(day: datetime, tz: str) -> list[tuple[str, str]]:
    # For mock purposes, generate every 30-minute slot 9am-5pm
    slots = []
    start_dt = day.replace(hour=9, minute=0, second=0, microsecond=0)
    end_dt = day.replace(hour=17, minute=0, second=0, microsecond=0)
    cur = start_dt
    while cur < end_dt:
        slots.append(
            (
                cur.isoformat(timespec="seconds"),
                (cur + timedelta(minutes=30)).isoformat(timespec="seconds"),
            )
        )
        cur += timedelta(minutes=30)
    return slots


def main(user_prompt: str):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    client = OpenAI(api_key=OPENAI_API_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    system_prompt = """
You are TriageBot. Your job is to triage unread email threads and draft replies. Steps:
1) Fetch recent unread threads.
2) Identify which threads imply a meeting request (based on subject/snippet).
3) Fetch calendar availability for the next 7 days during business hours.
4) For each meeting-request thread, propose 2 to 3 specific 30-minute slots that are free, and draft a concise reply with those options, timezone, and a fallback ("send more times if these don't work").
5) For non-meeting threads, draft a brief triage note or a one-line acknowledgment if appropriate.
Always run email fetching and calendar availability in parallel. Keep replies short and professional.

Parallelization guidance:
- Immediately call listUnreadThreads and getCalendarAvailability for the next 7 days in parallel; do not wait for one to finish before starting the other.
- Calendar fetching should not depend on knowing which threads need meetings; fetch once for the next 7 days and reuse for all meeting-related threads.
- If multiple meeting-request threads are found, generate proposed slots and draft replies independently for each thread.

Output requirements:
- Summary: count of unread threads scanned and how many need meetings.
- For each meeting thread: sender, subject, 2 to 3 proposed 30-minute free slots (include dates/times and timezone), and a concise reply draft that proposes those slots and invites alternatives.
- For non-meeting threads: one-line triage suggestion (e.g., archive, quick acknowledgment, or follow-up needed).
- Be explicit about timezone in proposed times and avoid outside-business-hours slots (default 9am to 5pm local unless the tool provides a different window).
"""

    tools = [
        {
            "type": "function",
            "name": "listUnreadThreads",
            "description": "Get recent unread email threads with lightweight metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Optional limit, default 10",
                    }
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "getCalendarAvailability",
            "description": "Get free/busy slots for the given time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "range": {
                        "type": "object",
                        "properties": {
                            "startIso": {"type": "string"},
                            "endIso": {"type": "string"},
                        },
                        "required": ["startIso", "endIso"],
                    }
                },
                "required": ["range"],
            },
        },
    ]

    # Mock data for deterministic behavior
    unread_threads = [
        {
            "id": "t1",
            "from": "alex@example.com",
            "subject": "Quick sync this week?",
            "snippet": "Are you free for a 30-min chat to discuss the roadmap?",
            "needsMeeting": True,
        },
        {
            "id": "t2",
            "from": "billing@example.com",
            "subject": "Invoice attached",
            "snippet": "Please review the attached invoice.",
            "needsMeeting": False,
        },
        {
            "id": "t3",
            "from": "sam@example.com",
            "subject": "Meet next week about launch",
            "snippet": "Can we find time next week?",
            "needsMeeting": True,
        },
    ]

    # Calendar availability mock: generate business-hour slots for next 7 days,
    # mark some as busy for variety
    now = datetime.now()
    tz = "America/Los_Angeles"
    start_iso, end_iso = next_7_day_range_iso(now)

    all_slots = []
    for d in range(0, 7):
        day = (now + timedelta(days=d)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for s in business_hours_slots(day, tz):
            all_slots.append({"startIso": s[0], "endIso": s[1], "busy": False})
    # Mark every 4th slot busy as a simple pattern
    for i in range(0, len(all_slots), 4):
        all_slots[i]["busy"] = True

    def call_tool(name: str, args: dict):
        if name == "listUnreadThreads":
            limit = int(args.get("limit", 10))
            return {"threads": unread_threads[:limit]}
        elif name == "getCalendarAvailability":
            _range = args["range"]
            # ignore provided range and return deterministic mock based on now
            return {"slots": all_slots, "tz": tz}
        else:
            raise ValueError(f"Unknown tool: {name}")

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

    tool_outputs = []
    round_num = 1
    while True:
        messages += response.output
        current_len = len(messages)

        for tool_call in response.output:
            if tool_call.type != "function_call":
                continue
            name = tool_call.name
            args = json.loads(tool_call.arguments)
            logger.info(f"Tool requested (round {round_num}): {name} {args}")

            result = call_tool(name, args)
            tool_outputs.append(
                {
                    "name": name,
                    "args": args,
                    "result": (
                        list(result.keys()) if isinstance(result, dict) else result
                    ),
                }
            )

            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(result),
                }
            )

        if len(messages) == current_len:
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
        "tool_outputs_keys": tool_outputs,
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
        "Triage my inbox. Check unread threads and, if any are asking to meet this week, propose 2 to 3 30-minute slots over the next 7 days and draft replies. Keep replies concise and include my timezone."
    )
