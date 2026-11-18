"""The agent: plan -> choose tools -> retrieve / refine -> answer with sources."""
import json
import re

from .store import KnowledgeBase
from .tools import analyze_data, calculate

DEFAULT_MODEL = "claude-sonnet-5-5"

TOOLS = [
    {"name": "list_sources",
     "description": "List the documents in the knowledge base (name, chunk count, whether it is a CSV table).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "search_knowledge_base",
     "description": "Semantic search over the knowledge base. Returns numbered passages with similarity scores. "
                    "Use short, focused queries. Call it again with a rewritten query or a different sub-question "
                    "if the evidence is weak or incomplete. Optionally restrict to one source.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "source": {"type": "string", "description": "Optional exact source name to search inside"},
         "top_k": {"type": "integer", "description": "Passages to return (default 4, max 8)"}},
         "required": ["query"]}},
    {"name": "summarize_source",
     "description": "Summarize one whole document, optionally focused on a topic. Use for overview questions.",
     "input_schema": {"type": "object", "properties": {
         "source": {"type": "string"}, "focus": {"type": "string"}}, "required": ["source"]}},
    {"name": "analyze_data",
     "description": "Run exact analysis on a CSV table in the knowledge base. operation: describe | head | value_counts "
                    "| stat | groupby | top. Use this instead of reading rows when you need counts, sums, averages, rankings.",
     "input_schema": {"type": "object", "properties": {
         "source": {"type": "string"},
         "operation": {"type": "string", "enum": ["describe", "head", "value_counts", "stat", "groupby", "top"]},
         "column": {"type": "string"}, "group_by": {"type": "string"},
         "agg": {"type": "string", "enum": ["mean", "sum", "min", "max", "median", "count"]},
         "n": {"type": "integer"}, "ascending": {"type": "boolean"}},
         "required": ["source", "operation"]}},
    {"name": "calculate",
     "description": "Evaluate an arithmetic expression exactly (+ - * / ** %, sqrt, round, min, max, sum). "
                    "Always use this for arithmetic instead of computing in your head.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
]

PLAN_PROMPT = """You are the planning step of an agentic RAG system over a private knowledge base.
Knowledge base sources: {sources}

Given the user's question, reply with ONLY a JSON object:
{{"complexity": "simple" | "multi_step" | "no_retrieval",
  "strategy": "<one sentence: how you will answer>",
  "sub_questions": ["<atomic question to look up>", ...],
  "tools": ["<tools you expect to need>"]}}
Use "no_retrieval" for greetings or questions that need no documents. Use "multi_step" when the answer needs
several lookups, comparison, filtering or arithmetic. Keep sub_questions to at most 4."""

SYSTEM = """You are the reasoning agent of a document-grounded assistant. You answer ONLY from the knowledge base.

How to work:
1. Follow the plan, but adapt it to what you find.
2. Retrieve with focused queries. Judge the results: are they relevant (scores below ~0.45 are usually weak) and
   sufficient? If not, rewrite the query (synonyms, narrower terms, a different source) and search again.
3. For counts, sums, averages or rankings over a CSV table use analyze_data; for any arithmetic use calculate.
4. Compare information across sources when the question asks for it, and say when sources disagree.
5. Cite every factual claim with the passage numbers you were given, like [1] or [2][3]. Numbers come from search
   results only. Never invent a citation. Facts that come from analyze_data should cite the table's passage.
6. If the knowledge base does not contain the answer, say so plainly and say what you did look for. Do not guess.
7. Final answer: clear and direct, no preamble about your process. Short paragraphs or a short list when it helps.

Knowledge base sources: {sources}

Plan from the planning step:
{plan}"""

STANDARD_SYSTEM = ("Answer the question using ONLY the numbered context passages. Cite passages like [1]. "
                   "If the context does not contain the answer, say so.")


class AgenticRAG:
    def __init__(self, kb: KnowledgeBase, client, model: str = DEFAULT_MODEL, max_steps: int = 6, top_k: int = 4):
        self.kb, self.client, self.model = kb, client, model
        self.max_steps, self.top_k = max_steps, top_k
        self.evidence = {}      # citation number -> {"source","page","text","score"}
        self._chunk_no = {}     # chunk id -> citation number
        self._viz = None        # what the last tool call looked like on the terrain map

    # ---------- LLM helpers ----------
    def _llm(self, system, messages, max_tokens=1200, tools=None):
        kw = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages)
        if tools:
            kw["tools"] = tools
        return self.client.messages.create(**kw)

    @staticmethod
    def _text(resp):
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    # ---------- planning ----------
    def plan(self, question: str) -> dict:
        try:
            resp = self._llm(PLAN_PROMPT.format(sources=", ".join(self.kb.sources()) or "(empty)"),
                             [{"role": "user", "content": question}], max_tokens=400)
            raw = self._text(resp)
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            data.setdefault("complexity", "simple")
            return data
        except Exception:
            return {"complexity": "simple", "strategy": "Search the knowledge base and answer from the results.",
                    "sub_questions": [question], "tools": ["search_knowledge_base"]}

    # ---------- evidence ----------
    def _cite(self, chunk, score) -> int:
        if chunk.id not in self._chunk_no:
            n = len(self._chunk_no) + 1
            self._chunk_no[chunk.id] = n
            self.evidence[n] = {"source": chunk.source, "page": chunk.page, "text": chunk.text, "score": score,
                                "chunk_id": chunk.id}
        return self._chunk_no[chunk.id]

    # ---------- tool execution ----------
    def _exec(self, name: str, args: dict) -> str:
        self._viz = None
        try:
            if name == "list_sources":
                return "\n".join(f"- {s} ({self.kb.count(s)} chunks{', table' if s in self.kb.tables else ''})"
                                 for s in self.kb.sources()) or "Knowledge base is empty."
            if name == "search_knowledge_base":
                k = max(1, min(int(args.get("top_k") or self.top_k), 8))
                hits = self.kb.search(args["query"], k, args.get("source"))
                if not hits:
                    return "No passages found."
                self._viz = {"kind": "search", "q": args["query"], "xy": self.kb.project(self.kb.last_q),
                             "hits": [[c.id, round(s, 3)] for c, s in hits], "best": round(hits[0][1], 3)}
                lines = [f"best_score={hits[0][1]:.2f}"]
                for chunk, score in hits:
                    n = self._cite(chunk, score)
                    page = f", p.{chunk.page}" if chunk.page else ""
                    lines.append(f"[{n}] ({chunk.source}{page}, score {score:.2f})\n{chunk.text}")
                return "\n\n".join(lines)
            if name == "summarize_source":
                text = self.kb.source_text(args["source"])
                if not text:
                    return f"Unknown source '{args['source']}'. Use list_sources."
                focus = f" Focus on: {args['focus']}." if args.get("focus") else ""
                resp = self._llm("Summarize the document in at most 150 words, factual and concise." + focus,
                                 [{"role": "user", "content": text}], max_tokens=400)
                # register the first chunk of the source so the summary can be cited
                first = next(c for c in self.kb.chunks if c.source == args["source"])
                n = self._cite(first, 1.0)
                self._viz = {"kind": "tool", "name": "summarize", "source": args["source"]}
                return f"[{n}] Summary of {args['source']}:\n{self._text(resp)}"
            if name == "analyze_data":
                df = self.kb.tables.get(args["source"])
                if df is None:
                    return f"'{args['source']}' is not a CSV table. Tables: {list(self.kb.tables) or 'none'}"
                out = analyze_data(df, args["operation"], args.get("column"), args.get("group_by"),
                                   args.get("agg", "mean"), int(args.get("n") or 5), bool(args.get("ascending", False)))
                first = next(c for c in self.kb.chunks if c.source == args["source"])
                self._viz = {"kind": "tool", "name": "analyze", "source": args["source"]}
                return f"[{self._cite(first, 1.0)}] {args['source']} analysis:\n{out}"
            if name == "calculate":
                self._viz = {"kind": "calc", "label": str(args["expression"])[:34]}
                return calculate(args["expression"])
            return f"Unknown tool {name}"
        except Exception as e:
            return f"Tool error: {e}"

    @staticmethod
    def _short(name: str, out: str) -> str:
        if name == "search_knowledge_base":
            m = re.search(r"best_score=([\d.]+)", out)
            n = len(re.findall(r"^\[\d+\]", out, re.M))
            return f"{n} passages, best score {m.group(1)}" if m else out[:120]
        return out.replace("\n", " ")[:140]

    # ---------- main loop (generator of trace events) ----------
    def run(self, question: str, history: list[dict] | None = None):
        self.evidence, self._chunk_no = {}, {}
        plan = self.plan(question)
        yield {"type": "plan", "plan": plan}

        messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:]]
        messages.append({"role": "user", "content": question})
        system = SYSTEM.format(sources=", ".join(self.kb.sources()) or "(empty)", plan=json.dumps(plan, indent=1))
        final, calls = "", 0

        for _ in range(self.max_steps):
            resp = self._llm(system, messages, max_tokens=1500, tools=TOOLS)
            uses = [b for b in resp.content if b.type == "tool_use"]
            text = self._text(resp)
            if not uses:
                final = text
                break
            if text:
                yield {"type": "thought", "text": text}
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for u in uses:
                calls += 1
                yield {"type": "tool_call", "name": u.name, "input": dict(u.input)}
                out = self._exec(u.name, dict(u.input))
                yield {"type": "observation", "name": u.name, "summary": self._short(u.name, out), "viz": self._viz}
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
            messages.append({"role": "user", "content": results})
        else:
            messages.append({"role": "user", "content": "Step limit reached. Answer now using the evidence gathered, "
                                                        "and say what is missing if it is incomplete."})
            final = self._text(self._llm(system, messages, max_tokens=1200))

        src = self.cited(final)
        yield {"type": "answer", "text": final, "sources": src, "tool_calls": calls,
               "cited_ids": [s["chunk_id"] for s in src]}

    def cited(self, text: str) -> list[dict]:
        nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)} & self.evidence.keys())
        return [{"n": n, **self.evidence[n]} for n in nums]

    # ---------- baseline for comparison ----------
    def standard(self, question: str) -> dict:
        self.evidence, self._chunk_no = {}, {}
        hits = self.kb.search(question, self.top_k)
        viz = {"q": question, "xy": self.kb.project(self.kb.last_q), "hits": [[c.id, round(s, 3)] for c, s in hits]} if hits else None
        ctx = "\n\n".join(f"[{self._cite(c, s)}] ({c.source}) {c.text}" for c, s in hits)
        resp = self._llm(STANDARD_SYSTEM, [{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}])
        text = self._text(resp)
        return {"text": text, "sources": self.cited(text), "viz": viz}
