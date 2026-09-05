# Data Agent — Agentic AI Platform

A multi-agent system built on LangGraph that routes natural-language requests to two specialized sub-agents: a **SQL Analyst** for database questions, and an **ETL Analyst** for data extraction and transformation. A router node classifies intent and dispatches accordingly — the user never needs to know which sub-agent is handling their request.

## Architecture

```
                       Data Agent (Router)
                 classifies intent -> sql | etl
                    │                    │
                    ▼                    ▼
            SQL Analyst Agent     ETL Analyst Agent
```

**SQL Analyst** — curate question → gather schema context → generate SQL → safety check (loops back on failure) → execute on Postgres → return answer

**ETL Analyst** — bind tools to LLM → interpret transformation intent → select tool (`extract_load_tool` / `transform_load_tool`) → generate Pandas code → execute safely → report result

LLM selection is dynamic: simpler queries route to a lighter/cheaper model, complex ones to high model.

## Features

- Intent-based routing between SQL and ETL workflows, no manual mode-switching
- Natural-language-to-SQL generation with a safety gate — destructive statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`) are blocked before execution, and results are capped at 10 rows by default
- API extraction and Pandas-based transformation, output to CSV, JSON, or Parquet
- Cost-aware LLM routing — cheaper models for simple queries, high models for complex reasoning
- Typed state throughout, via Pydantic schemas for each agent

## Prerequisites

- Python 3.12+
- PostgreSQL instance
- Anthropic and/or OpenAI API key

## Installation

```bash
git clone https://github.com/Bindukasetty/AutoData_Agent.git
cd Ola AI Project
python -m venv .venv
source .venv/bin/activate   # .\.venv\Scripts\Activate.ps1 on Windows

pip install -e .
# or: uv pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key

host=localhost
port=5432
user=postgres
password=your_password
database=data_agent_db

LLM_MODEL_LOW=gpt-5.6-luna
LLM_MODEL_MEDIUM=gpt-5.6-terra
LLM_MODEL_HIGH=gpt-5.6-sol
```

## Project structure

```
Ola AI Project/
├── agents/
│   ├── data_agent.py       # Router
│   ├── sql_analyst.py      # SQL workflow
│   └── etl_analyst.py      # ETL workflow
├── Models/
│   └── schema.py           # Pydantic state schemas
├── utils/
│   ├── database.py         # Postgres connection + schema introspection
│   ├── etl_tools.py        # extract_load_tool, transform_load_tool
│   └── llm_pick.py         # complexity-based model selection
├── data/
│   ├── extract/
│   ├── transform/
│   └── *.csv                # sample datasets
├── main.py
├── feed_db.py
└── pyproject.toml
```

## Usage

```python
from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage

# Routed to SQL Analyst
data_agent.invoke({
    "messages": [HumanMessage(content="Show me the top 5 users with the highest ratings")],
    "route_response": ""
})

# Routed to ETL Analyst
data_agent.invoke({
    "messages": [HumanMessage(content=
        "Extract data from https://pokeapi.co/api/v2/pokemon and save it to data/extract as CSV"
    )],
    "route_response": ""
})
```

```bash
python main.py
```

## State schemas

| Schema | Purpose |
|---|---|
| `DataAgentSchema` | Top-level state: messages + router decision |
| `RouterSchema` | Classification output: `sql` \| `etl` + reasoning |
| `AgentSchema` | SQL agent state: curated question, generated SQL, safety flag, execution result, final answer |
| `ETLAgentSchema` | ETL agent state: messages/tool-call history |

## Design notes

- **Safety before execution**: the SQL agent's safety check isn't a filter after the fact — an unsafe query is rejected and re-generated with the rejection reason fed back to the LLM, so the loop self-corrects instead of just failing.
- **Tool-based ETL, not a fixed pipeline**: the ETL agent reasons about which tool to call rather than following a hardcoded extract-transform-load sequence, so adding a new source is a new tool, not a new pipeline.
- **Cost-aware routing**: query complexity determines model choice, so simple lookups don't pay for a premium model.

