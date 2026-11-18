# Pathfinder: Agentic RAG System

An agent that plans before answering, decides when and how to retrieve, rewrites weak searches, uses tools
(search, summarization, data analysis, calculator), and answers only from your documents, with sources.
Built for the Instant AI & Data Analysis training (Sprint 4).

## Standard RAG vs Agentic RAG

| | Standard RAG | Agentic RAG (this project) |
|---|---|---|
| Retrieval | one search with the raw question | the agent decides if, when and how often |
| Weak results | answers anyway | rewrites the query and searches again |
| Multi-step questions | struggles | splits into sub-questions, compares sources |
| Numbers | LLM guesses | `analyze_data` and `calculate` tools |
| Cost / latency | low | higher (several model calls) |

Turn on **Compare with Standard RAG** in the sidebar to see both answers side by side.

## The Terrain

The top of the app is a live map of your knowledge base: every passage is a dot, positioned by a 2-D projection
of its embedding, so related passages sit close together and each document forms a territory.

When you ask a question, the agent's route is drawn across the map in real time:

- a **ring** marks each search, with the query as its label; lines fan out to the passages it retrieved
- weak searches (low similarity) glow **coral**, and you can watch the agent hop to a rewritten query
- **diamonds** and **Σ** mark analysis, summarization and calculator calls
- the **answer flag** lands on the centre of the cited passages, which light up with white rings
- with *Compare with Standard RAG* on, a grey dashed **one-shot** probe shows how little ground standard RAG covers
- hover any dot to read the passage; use *Replay route* under any answer to watch it again

## Pipeline

```
upload (PDF/DOCX/CSV/TXT/MD) -> extract -> clean -> chunk (900 chars, 150 overlap)
   -> embed (bge-small, ONNX) -> in-memory vector store
question -> PLAN (JSON) -> agent loop (max N steps):
      search_knowledge_base | summarize_source | analyze_data | calculate | list_sources
   -> grounded answer with [n] citations -> sources panel + full agent trace
```

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

Press **Load samples** in the sidebar for a ready-made knowledge base. Try:

- "Which course is the longest and who teaches it?"
- "What would the two most expensive AI courses cost for a group of 3 with the company discount?"
- "Compare the refund policy before and after day 14, and summarize the handbook."
- "What is the capital of France?" (should say it isn't in the knowledge base)

## Deploy on Streamlit Cloud

1. Push this folder to GitHub.
2. New app on share.streamlit.io, main file `app.py`.
3. App settings, Secrets: `ANTHROPIC_API_KEY = "sk-ant-..."`

## Layout

```
app.py            Streamlit UI (chat, live agent trace, sources, comparison)
rag/ingest.py     extraction, cleaning, chunking
rag/store.py      embeddings + vector search
rag/tools.py      calculator, table analysis
rag/agent.py      planner, tool loop, standard-RAG baseline
rag/mapview.py    the Terrain: canvas map + animated route
tests/            offline smoke test (python tests/test_smoke.py)
```
