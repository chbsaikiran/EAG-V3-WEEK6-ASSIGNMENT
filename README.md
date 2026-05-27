# Agent6 — Memory · Perception · Decision · Action

A cognitive agent built on top of **LLM Gateway V3**. The agent decomposes any user query into goals, executes them step by step using real web tools, stores results in a persistent memory file,  and produces a final answer — all without keeping any state inside the LLM.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        agent6/                              │
│                                                             │
│   User Query                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐    read()    ┌──────────────┐                 │
│  │ Memory  │◄────────────►│ memory.json  │  (persists      │
│  │ Module  │    write()   │  (facts,     │   across runs)  │
│  └────┬────┘              │  prefs,      │                 │
│       │ hits              │  outcomes)   │                 │
│       ▼                   └──────────────┘                 │
│  ┌────────────┐                                             │
│  │ Perception │  Gemini (pinned)  ← goals decomposed once  │
│  │  Module    │  marks done/open each iteration             │
│  └─────┬──────┘                                             │
│        │ current_goal + attach_artifact_id?                 │
│        ▼                                                     │
│  ┌────────────┐  auto_route="decision"                      │
│  │  Decision  │  (Router picks TINY/LARGE tier)             │
│  │  Module    │  → tool_call  OR  answer                    │
│  └─────┬──────┘                                             │
│        │ tool_call                                          │
│        ▼                                                     │
│  ┌────────────┐  MCP stdio                                  │
│  │   Action   │──────────► mcp_server.py (9 tools)          │
│  │  Module    │◄──────────  result text                      │
│  └─────┬──────┘                                             │
│        │ > 4 KB?                                            │
│        ▼                                                     │
│  ┌──────────────┐                                           │
│  │ Artifact     │  state/artifacts/<sha256>.bin             │
│  │ Store        │  art:<handle> stored in memory.json       │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

### The Four Roles

| Role | File | LLM | Purpose |
|---|---|---|---|
| **Memory** | `memory.py` | None (pure Python) | Read/write `state/memory.json` — keyword overlap search, record tool outcomes, store facts |
| **Perception** | `perception.py` | Gemini (pinned) | Decompose query → goals; mark done; attach artifact indices |
| **Decision** | `decision.py` | auto_route="decision" | Choose next action: call one MCP tool OR give a final answer |
| **Action** | `action.py` | None | Dispatch MCP tool call; store >4 KB results as artifacts |

### Key Design Decisions

| What | Why |
|---|---|
| Perception pinned to Gemini | Smaller models drop goals, hallucinate artifact indices, and break goal identity across iterations |
| Goals identified by **position**, not by string id | The LLM cannot invent a stale id; the outer loop owns all stable ids |
| Artifact references use **integer index** into MEMORY HITS | Model picks from indices it actually sees — no free-text hallucination of art: handles |
| `auto_route="decision"` on Decision | Router picks TINY for short goals, LARGE for artifact-heavy ones; Gemini fallback for HUGE contexts |
| `art:` guard in `action.py` | Blocks models from passing artifact handles as file paths or URLs to tools |
| All LLM calls are **stateless** | The agent maintains all state (history, goals, artifacts) — not the LLM |
| Tool results >4 KB stored as artifacts | Keeps history compact; raw bytes attached to Decision only when Perception decides they are needed |

---

## File Structure

```
Assignment/
├── agent6/
│   ├── __init__.py          # package entry: run(), main()
│   ├── __main__.py          # python -m agent6 "query"
│   ├── schemas.py           # Pydantic models: MemoryItem, Goal, Observation,
│   │                        #   AgentToolCall, DecisionOutput, Artifact
│   ├── memory.py            # pure-Python read / filter / record_outcome / remember
│   ├── artifact_store.py    # SHA256-keyed binary store for >4 KB payloads
│   ├── perception.py        # Gemini-pinned goal orchestrator
│   ├── decision.py          # auto_route LLM call → tool or answer
│   ├── action.py            # pure MCP dispatch, ~35 lines, no LLM
│   └── agent.py             # main cognitive loop
│
├── mcp_server.py            # MCP server — 9 tools (web_search, fetch_url, …)
├── main.py                  # LLM Gateway V3 (FastAPI, port 8101)
├── providers.py             # Provider adapters (Gemini, Groq, NVIDIA, …)
├── router.py                # Failover rings and rate-state
├── schemas.py               # Gateway Pydantic models
├── client.py                # Python SDK for the gateway
├── .env                     # API keys (not committed)
│
└── state/                   # created at runtime
    ├── memory.json           # durable across runs
    └── artifacts/
        ├── index.json
        └── <sha256[:16]>.bin
```

---

## Agent Loop (one iteration)

```
memory.read(query, history)      → hits
perception.observe(query, hits, history, prior_goals)
    → Observation(goals with done flags + attach_artifact_id)
[all goals done?] → exit
load artifact bytes if attach_artifact_id is set
decision.next_step(goal, hits, attached, history, mcp_tools)
    → DecisionOutput(answer | tool_call)
if tool_call:
    action.execute(session, tool_call)   → (descriptor, artifact_id | None)
    memory.record_outcome(...)
    history.append(tool result)
if answer:
    history.append(answer)
repeat
```

---

## Setup

### 1. Prerequisites

The project uses the `.venv` in the Assignment folder. All dependencies are already installed there.

```bat
.venv\Scripts\activate
```

### 2. `.env` file

Create `.env` in the Assignment folder:

```env
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here        # for web_search

# Optional
GEMINI_MODEL=gemini-2.0-flash
GROQ_API_KEY=...
NVIDIA_API_KEY=...
GATEWAY_V3_PORT=8101
```

### 3. Start the gateway (Terminal 1)

```bat
.venv\Scripts\python.exe main.py
```

Verify it has providers: http://localhost:8101

### 4. Run the agent (Terminal 2)

```bat
uv run python -m agent6 "Your query here"
```

> **Note:** `uv run python` resolves to the system Python but the agent auto-detects `.venv\Scripts\python.exe` for the MCP subprocess. If you see import errors, activate the venv first and run `python -m agent6 "..."` directly.

---

## MCP Tools (from `mcp_server.py`)

| Tool | Description |
|---|---|
| `web_search` | Tavily primary, DuckDuckGo fallback. Hard-capped at 5 results |
| `fetch_url` | Clean markdown via crawl4ai (headless Chromium) |
| `get_time` | Current time in any IANA timezone |
| `currency_convert` | Live rates via frankfurter.dev |
| `read_file` | Read a file from the sandbox |
| `list_dir` | List a sandbox directory |
| `create_file` | Create a new file (parent directory must exist) |
| `update_file` | Overwrite an existing sandbox file |
| `edit_file` | Find-and-replace inside a sandbox file |

Files are sandboxed under `./sandbox/`. Subdirectories `reminders/`, `notes/`, `research/` are pre-created by the agent on startup.

---

## The Four Test Queries

### Query A — Artifact Attach Test

```bat
uv run python -m agent6 "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
```

**What it exercises:** The full artifact attach path. The web search result (~6 KB) is stored as an artifact. Perception sees the artifact handle in memory hits and sets `attach_artifact_id` on the extraction goal. Decision reads the raw bytes and answers in one call.

```
─── iter 1 ───
[memory.read]   0 hits
[perception]    [open] Fetch the Wikipedia page for Claude Shannon.
[perception]    [open] Extract his birth date, death date, and three key contributions to information theory from the fetched content.
[decision]      TOOL_CALL: web_search({"query": "Wikipedia Claude Shannon"})
[action]        → [artifact art:dca38a7ce9199feb, 6,283 bytes] preview: { "title": "Claude Shannon - Wikipedia" ...

─── iter 2 ───
[memory.read]   1 hits
                tool_outcome: web_search(query=Wikipedia Claude Shannon) → art:dca38a7ce9199feb  [artifact: art:dca38a7ce9199feb]
[perception]    [done] Fetch the Wikipedia page for Claude Shannon.
[perception]    [open] Extract his birth date, death date, and three key contributions to information theory from the fetched content.
                  attach=art:dca38a7ce9199feb
[attach]        art:dca38a7ce9199feb (6,283 bytes)
[decision]      router/503 → fallback to gemini direct
[decision]      ANSWER: Claude Shannon was born on April 30, 1916, and passed away on February 24, 2001. His three key contributions ...

─── iter 3 ───
[memory.read]   1 hits
[perception]    [done] Fetch the Wikipedia page for Claude Shannon.
[perception]    [done] Extract his birth date, death date, and three key contributions to information theory from the fetched content.
                  attach=art:dca38a7ce9199feb

[done] all 2 goals satisfied

FINAL: Claude Shannon was born on April 30, 1916, and passed away on February 24, 2001. His three key
contributions to information theory include:

1. **Founding the Field:** He established the mathematical foundations of information theory with his 1948
   paper, "A Mathematical Theory of Communication," which is widely regarded as the "Magna Carta of the
   Information Age."
2. **Defining the "Bit":** He introduced the concept of measuring information using binary digits, or "bits,"
   providing a quantitative way to define and measure information as the resolution of uncertainty.
3. **The Noisy-Channel Coding Theorem:** He formulated this fundamental theorem, which proved that it is
   possible to transmit information over a noisy channel with an arbitrarily low probability of error,
   provided the transmission rate is below the channel's capacity.
```

---

### Query B — Multi-Goal + Weather Constraint

```bat
uv run python -m agent6 "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."
```

**What it exercises:** Three sequential goals — search activities, search weather, cross-reference and recommend. Memory carries the weather artifact handle from iteration 2 into Decision's context in iteration 3.

```
─── iter 1 ───
[memory.read]   1 hits
[perception]    [open] Search for 3 family-friendly activities in Tokyo suitable for the weekend.
[perception]    [open] Search for the weather forecast in Tokyo for this Saturday.
[perception]    [open] Recommend the most appropriate activity based on the weather forecast.
[decision]      TOOL_CALL: web_search({"query": "family-friendly activities in Tokyo for the weekend"})
[action]        → [artifact art:b401a910d4b7718a, 8,870 bytes] preview: ...

─── iter 2 ───
[memory.read]   2 hits
[perception]    [done] Search for 3 family-friendly activities in Tokyo suitable for the weekend.
                  attach=art:b401a910d4b7718a
[perception]    [open] Search for the weather forecast in Tokyo for this Saturday.
[perception]    [open] Recommend the most appropriate activity based on the weather forecast.
[decision]      router/503 → fallback to gemini direct
[decision]      TOOL_CALL: web_search({"query": "weather forecast Tokyo this Saturday"})
[action]        → [artifact art:00c3ba6de460b777, 9,413 bytes] preview: ...

─── iter 3 ───
[memory.read]   3 hits
[perception]    [done] Search for 3 family-friendly activities in Tokyo suitable for the weekend.
[perception]    [done] Search for the weather forecast in Tokyo for this Saturday.
[perception]    [open] Recommend the most appropriate activity based on the weather forecast.
                  attach=art:b401a910d4b7718a
[attach]        art:b401a910d4b7718a (8,870 bytes)
[decision]      ANSWER: Given the weather forecast of "patchy rain" for this Saturday in Tokyo ...

─── iter 4 ───
[perception]    [done] Search for 3 family-friendly activities in Tokyo suitable for the weekend.
[perception]    [done] Search for the weather forecast in Tokyo for this Saturday.
[perception]    [done] Recommend the most appropriate activity based on the weather forecast.

[done] all 3 goals satisfied

FINAL: Given the weather forecast of "patchy rain" for this Saturday in Tokyo, I recommend planning indoor
activities to ensure your family remains comfortable and dry. Here are three excellent indoor options:

1. **Character Cafés:** Tokyo is famous for its themed dining experiences. Visiting a Pokémon or Kirby café
   provides a fun, immersive environment for children that is completely shielded from the weather.
2. **Interactive Workshops:** You might consider booking an authentic sushi-making class or a traditional
   tea ceremony in Asakusa. These hands-on cultural experiences are excellent for families and are held in
   climate-controlled indoor settings.
3. **Arcades and Entertainment Centers:** Areas like Akihabara are home to multi-story arcades that offer
   a variety of games and entertainment, making them a perfect "escape" from rainy conditions while still
   providing plenty of excitement for kids and adults alike.
```

---

### Query C — Durable Memory Across Two Runs

**Run 1:**
```bat
uv run python -m agent6 "My mom's birthday is 7 August 2026. Remember that and give me a calendar reminder for two weeks before and on the day."
```

**Run 2:**
```bat
uv run python -m agent6 "When is mom's birthday?"
```

**What it exercises:** `memory.remember()` is called at the very start of run 1, storing the date in `state/memory.json`. Run 2 reads the same file without any tool calls — the fact surfaces immediately through the keyword search and Decision answers directly from memory.

**Run 1:**
```
[memory.remember]  stored as fact: My mom's birthday is 7 August 2026. Remember that...
                   keywords: ['remember', 'give', 'calendar', 'weeks', 'august']

─── iter 1 ───
[memory.read]   2 hits
                fact: My mom's birthday is 7 August 2026. Remember that...
[perception]    [open] Store the date of mom's birthday as 7 August 2026.
[perception]    [open] Calculate the date two weeks before 7 August 2026.
[perception]    [open] Create calendar reminders for 24 July 2026 and 7 August 2026.
[decision]      TOOL_CALL: create_file({"path": "moms_birthday.txt", "content": "Mom's birthday: 7 August 2026.\nReminder 1: 24 July 2026 (two weeks before)..."})
[action]        → {"ok": true, "path": "moms_birthday.txt", "size_bytes": 117}

─── iter 2 ───
[perception]    [done] Store the date of mom's birthday as 7 August 2026.
[perception]    [done] Calculate the date two weeks before 7 August 2026.
[perception]    [done] Create calendar reminders for 24 July 2026 and 7 August 2026.

[done] all 3 goals satisfied
```

**Run 2:**
```
─── iter 1 ───
[memory.read]   3 hits
                fact: When is mom's birthday?
                fact: My mom's birthday is 7 August 2026. Remember that...
                tool_outcome: create_file(path=moms_birthday.txt, ...)
[perception]    [open] Identify the date of mom's birthday from the provided memory.
[perception]    [open] Confirm the creation of the calendar reminders for 24 July and 7 August.
[decision]      ANSWER: Mom's birthday is on 7 August 2026. This date was confirmed in the provided
                memory and has been documented in the file "moms_birthday.txt" within the system.
                You have also scheduled reminders for two weeks prior, on 24 July, and on the day itself.

─── iter 2 ───
[perception]    [done] Identify the date of mom's birthday from the provided memory.
[perception]    [done] Confirm the creation of the calendar reminders for 24 July and 7 August.

[done] all 2 goals satisfied

FINAL: Mom's birthday is on 7 August 2026. This date was confirmed in the provided memory and has been
documented in the file "moms_birthday.txt" within the system. You have also scheduled reminders for two
weeks prior, on 24 July, and on the day of the birthday itself.
```

---

### Query D — Multi-Source Synthesis

```bat
uv run python -m agent6 "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."
```

**What it exercises:** Web search produces a results artifact. Perception attaches it to the "read top 3 results" goal. Decision synthesizes a consolidated answer from the attached artifact content in a single call.

```
─── iter 1 ───
[memory.read]   5 hits
[perception]    [open] Search for 'Python asyncio best practices'.
[perception]    [open] Read the content of the top 3 search results.
[perception]    [open] Synthesize a numbered list of the common advice found across the top 3 results.
[decision]      TOOL_CALL: web_search({"query": "Python asyncio best practices"})
[action]        → [artifact art:d29c2990ffc93419, 9,633 bytes] preview: ...

─── iter 2 ───
[memory.read]   6 hits
                tool_outcome: web_search(query=Python asyncio best practices) → art:d29c2990ffc93419  [artifact]
[perception]    [done] Search for 'Python asyncio best practices'.
[perception]    [open] Read the content of the top 3 search results.
                  attach=art:d29c2990ffc93419
[perception]    [open] Synthesize a numbered list of the common advice found across the top 3 results.
[attach]        art:d29c2990ffc93419 (9,633 bytes)
[decision]      ANSWER: Based on the search results provided, here are the key best practices ...

─── iter 3 ───
[perception]    [done] Search for 'Python asyncio best practices'.
[perception]    [open] Read the content of the top 3 search results.
                  attach=art:d29c2990ffc93419
[perception]    [open] Synthesize a numbered list of the common advice found across the top 3 results.
[attach]        art:d29c2990ffc93419 (9,633 bytes)
[decision]      router/503 → fallback to gemini direct
[decision]      ANSWER: The top three search results for asyncio best practices provide the following insights ...

─── iter 4 ───
[perception]    [done] Search for 'Python asyncio best practices'.
[perception]    [done] Read the content of the top 3 search results.
[perception]    [done] Synthesize a numbered list of the common advice found across the top 3 results.

[done] all 3 goals satisfied

FINAL: The top three search results for `asyncio` best practices provide the following insights:

1. **Asyncio best practices - Async-SIG - Discussions on Python.org:** This community discussion highlights
   that async functions are not inherently asynchronous and must be integrated with the `asyncio` library
   to achieve concurrency. It emphasizes avoiding long-running loops that block the event loop and suggests
   scheduling iterations onto the loop instead. It also discusses the importance of using `tasks` to track
   and manage concurrent operations.

2. **Python's asyncio: A Hands-On Walkthrough (Real Python):** This tutorial explains that `asyncio` is
   best suited for I/O-bound tasks where it can outperform multithreading by avoiding thread management
   overhead. It stresses that `asyncio.run()` should be the standard entry point for programs and warns
   that if tasks created with `create_task()` are not awaited or gathered, they will be canceled when the
   main coroutine finishes.

3. **Asyncio Best Practices and Common Pitfalls (Shane's Personal Blog):** This resource reinforces the
   necessity of using `asyncio.run()` to ensure proper setup and cleanup of the event loop. It advocates
   for the use of `async with` (async context managers) for efficient resource management and highlights
   the critical mistake of forgetting to `await` coroutines, which prevents them from executing as intended.
```

---

## Memory System

`state/memory.json` persists across all runs. Four item kinds:

| Kind | Written by | Contains |
|---|---|---|
| `fact` | `memory.remember()` at run start | User-stated facts ("mom's birthday is…") |
| `preference` | `memory.remember()` at run start | User preferences ("I prefer…") |
| `tool_outcome` | `memory.record_outcome()` after every Action | Tool name, arguments, result preview, artifact handle |
| `scratchpad` | (reserved for future use) | — |

**Read algorithm** (`memory.read`): tokenises query + last 4 history messages into a set of lowercase word tokens. Scores each memory item by `overlap / len(query_tokens)`. Returns top-k by score, falling back to most-recent items when no overlap is found.

---

## Artifact Store

`state/artifacts/` stores tool results larger than 4 KB.

- **Handle format:** `art:<sha256[:16]>` — deterministic, content-addressed
- **Index:** `state/artifacts/index.json` maps handle → metadata + file path
- **Attachment:** Perception sets `attach_artifact_id` on a goal when raw bytes are needed. The agent loop loads the bytes and embeds them in the Decision prompt under `ATTACHED ARTIFACTS:`.
- **Guard:** Action rejects any tool argument that starts with `art:` — handles are not file paths.

---

## LLM Gateway V3

The gateway (`main.py`, port 8101) sits between the agent and 7 LLM providers:

| Provider | Role |
|---|---|
| Gemini | Perception (pinned) + LARGE-tier Decision |
| Groq, NVIDIA, Cerebras | LARGE-tier Decision workers |
| GitHub Models, OpenRouter | TINY-tier Decision workers |
| Ollama | Local fallback |

**Router pool** (Cerebras, Groq, NVIDIA, GitHub with small models) classifies each Decision call as TINY or LARGE based on token count + content sample. When the prompt is too large for the router (HUGE, >8 000 tokens) or all providers are busy, Decision falls back to Gemini directly.

Dashboard: http://localhost:8101

---

## Prompt Evaluation Results

The system prompts for Perception and Decision were evaluated by an independent Prompt Evaluation Assistant against nine criteria for structured, step-by-step reasoning quality.

### Perception Prompt

| Criterion | Result |
|---|---|
| Explicit Reasoning Instructions | ✅ Yes |
| Structured Output Format | ✅ Yes |
| Separation of Reasoning and Tools | ✅ Yes |
| Conversation Loop Support | ✅ Yes |
| Instructional Framing | ✅ Yes |
| Internal Self-Checks | ✅ Yes |
| Reasoning Type Awareness | ❌ No |
| Error Handling / Fallbacks | ❌ No |

**Overall:** Excellent structure and strong operational constraints. The prompt clearly defines role behavior, completion criteria, output formatting, and multi-turn consistency. It strongly reduces hallucination through evidence-based completion checks and strict JSON output rules. It could be improved further by adding explicit reasoning-type labels (e.g., search vs extraction vs synthesis) and fallback behavior for ambiguous or missing run history.

---

### Decision Prompt

| Criterion | Result |
|---|---|
| Explicit Reasoning Instructions | ❌ No |
| Structured Output Format | ✅ Yes |
| Separation of Reasoning and Tools | ✅ Yes |
| Conversation Loop Support | ✅ Yes |
| Instructional Framing | ✅ Yes |
| Internal Self-Checks | ❌ No |
| Reasoning Type Awareness | ❌ No |
| Error Handling / Fallbacks | ✅ Yes |

**Overall:** Strong operational prompt with clear agent constraints, strict tool usage rules, and good multi-turn orchestration support. The prompt is highly robust against hallucinated actions and invalid tool calls through explicit restrictions on artifacts, arguments, and output behavior. However, it does not explicitly encourage step-by-step reasoning, self-verification, or reasoning-type labeling. Fallback handling is partially supported through guidance like calling `list_dir('.')` when uncertain about sandbox paths.
