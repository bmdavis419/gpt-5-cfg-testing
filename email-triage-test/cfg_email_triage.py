import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load secrets from project-level .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "email_triage_cfg.json"
LIST_THREADS_GRAMMAR_PATH = BASE_DIR / "list_unread_threads.lark"
GET_CAL_AVAIL_GRAMMAR_PATH = BASE_DIR / "get_calendar_availability.lark"

DEFAULT_MODEL_NAME = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "minimal"
MODEL_NAME = os.getenv("OPENAI_MODEL", DEFAULT_MODEL_NAME)
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)


def main(user_prompt: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    client = OpenAI(api_key=OPENAI_API_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with open(LIST_THREADS_GRAMMAR_PATH, "r") as f:
        list_threads_grammar = f.read()
    with open(GET_CAL_AVAIL_GRAMMAR_PATH, "r") as f:
        get_cal_avail_grammar = f.read()

    system_prompt = """
You are TriageBot. Your job is to triage unread email threads and draft replies. Steps:
1) Fetch recent unread threads.
2) Identify which threads imply a meeting request (based on subject/snippet).
3) Fetch calendar availability for the next 7 days during business hours.
4) For each meeting-request thread, propose 2–3 specific 30-minute slots that are free, and draft a concise reply with those options, timezone, and a fallback ("send more times if these don’t work").
5) For non-meeting threads, draft a brief triage note or a one-line acknowledgment if appropriate.
Always run email fetching and calendar availability in parallel. Keep replies short and professional.

Parallelization guidance:
- Immediately call listUnreadThreads and getCalendarAvailability for the next 7 days in parallel; do not wait for one to finish before starting the other.
- Calendar fetching should not depend on knowing which threads need meetings; fetch once for the next 7 days and reuse for all meeting-related threads.
- If multiple meeting-request threads are found, generate proposed slots and draft replies independently for each thread.

Output requirements:
- Summary: count of unread threads scanned and how many need meetings.
- For each meeting thread: sender, subject, 2–3 proposed 30-minute free slots (include dates/times and timezone), and a concise reply draft that proposes those slots and invites alternatives.
- For non-meeting threads: one-line triage suggestion (e.g., archive, quick acknowledgment, or follow-up needed).
- Be explicit about timezone in proposed times and avoid outside-business-hours slots (default 9am–5pm local unless the tool provides a different window).
"""

    tools = [
        {
            "type": "custom",
            "name": "listUnreadThreads",
            "description": "Get recent unread email threads with lightweight metadata.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": list_threads_grammar,
            },
        },
        {
            "type": "custom",
            "name": "getCalendarAvailability",
            "description": "Get free/busy slots for the given time range.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": get_cal_avail_grammar,
            },
        },
    ]

    # Mock data
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

    now = datetime.now()
    tz = "America/Los_Angeles"

    # Precompute availability for determinism
    def business_slots_for_next_7_days():
        def day_slots(day_dt: datetime):
            start = day_dt.replace(hour=9, minute=0, second=0, microsecond=0)
            end = day_dt.replace(hour=17, minute=0, second=0, microsecond=0)
            cur = start
            res = []
            while cur < end:
                res.append({
                    "startIso": cur.isoformat(timespec="seconds"),
                    "endIso": (cur + timedelta(minutes=30)).isoformat(timespec="seconds"),
                    "busy": False,
                })
                cur += timedelta(minutes=30)
            return res

        slots = []
        for d in range(0, 7):
            day = (now + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
            slots.extend(day_slots(day))
        for i in range(0, len(slots), 4):
            slots[i]["busy"] = True
        return slots

    all_slots = business_slots_for_next_7_days()

    def call_tool(name, args):
        if name == "listUnreadThreads":
            limit = int(args.get("limit", 10))
            return {"threads": unread_threads[:limit]}
        elif name == "getCalendarAvailability":
            # args contains: {"range": {"startIso": str, "endIso": str}}
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
            if tool_call.type != "custom_tool_call":
                continue
            name = tool_call.name
            args = json.loads(tool_call.input)
            logger.info(f"Tool requested (round {round_num}): {name} {args}")

            result = call_tool(name, args)
            tool_outputs.append({"name": name, "args": args, "result_keys": (list(result.keys()) if isinstance(result, dict) else result)})

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
    # Build the default 7-day range for the example call
    now = datetime.now()
    start_iso = now.isoformat(timespec="seconds")
    end_iso = (now + timedelta(days=7)).isoformat(timespec="seconds")
    main(
        "Triage my inbox. Check unread threads and, if any are asking to meet this week, propose 2–3 30-minute slots over the next 7 days and draft replies. Keep replies concise and include my timezone."
    )
