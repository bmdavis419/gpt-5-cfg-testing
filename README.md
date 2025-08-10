# GPT-5 CFG tool call testing

This is a pretty simple test of the GPT-5 CFG tool call functionality.

_this is not exhaustive and should not be taken as a comprehensive test of the GPT-5 CFG tool call functionality._

## FINDINGS

The weirdest thing I've been seeing is that whenever I use a CFG for my tool call, the model refuses to do tool calls in parallel. It will only do one tool call at a time.

I have zero clue why this is happening, below are screenshots of the terminal outputs with explanations.

### todos-test

**_THESE TESTS WERE DONE WITH GPT-5-MINI_**
I have been getting the same results with GPT-5 high reasoning effort.

**normal functions**

made 3 api calls to openai:

- 1 to get current date/time
- 1 to add 3 todos
- 1 for final summary

![normal functions](./images/todos-normal.png)

**cfg functions**

made 5 api calls to openai:

- 1 to get current date/time
- 1 to add 1 todo
- 1 to add 1 todo
- 1 to add 1 todo
- 1 for final summary

![cfg functions](./images/todos-cfg.png)

### price-test

**_THESE TESTS WERE DONE WITH GPT-5 HIGH REASONING EFFORT_**

**normal functions**

made 2 api calls to openai:

- 1 for 4 tool calls (price and shipping info)
- 1 for final summary

![price-normal](./images/price-normal.png)

**cfg functions**

made 5 api calls to openai:

- 1 to get price info tool call
- 1 to get shipping info tool call
- 1 to get price info tool call
- 1 to get shipping info tool call
- 1 for final summary

![price-cfg](./images/price-cfg.png)

### email-triage-test

**_THESE TESTS WERE DONE WITH GPT-5 MINIMAL REASONING EFFORT_**

**normal functions**

made 2 api calls to openai:

- 1 with 2 tool calls (list unread threads and get calendar availability)
- 1 for final summary

![email-triage-normal](./images/email-normal.png)

**cfg functions**

made 3 api calls to openai:

- 1 with 1 tool call (list unread threads)
- 1 with 1 tool call (get calendar availability)
- 1 for final summary

![email-triage-cfg](./images/email-cfg.png)

## GETTING STARTED

This project uses uv for package management and running.

1. get an openai api key and add it to a `.env` file in the root of the project

```
OPENAI_API_KEY=sk-...
```

2. install dependencies

```
uv sync
```

3. run the tests

- `uv run todos-test/cfg_functions.py`
- `uv run todos-test/normal_functions.py`
- `uv run price-test/cfg_price_compare.py`
- `uv run price-test/price_compare.py`
- `uv run email-triage-test/cfg_email_triage.py`
- `uv run email-triage-test/email_triage.py`

4. check the output in the `output` directory (and see what tools are being called per request in the terminal output)
