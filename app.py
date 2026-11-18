import os

import streamlit as st
import streamlit.components.v1 as components

from rag.agent import DEFAULT_MODEL, AgenticRAG
from rag.mapview import HEIGHT, build_map_html
from rag.store import Embedder, KnowledgeBase

st.set_page_config(page_title="Pathfinder · Agentic RAG", page_icon="🧭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=IBM+Plex+Mono:wght@400;500&display=swap');
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.02em; }
.hero .kicker { font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.22em; color:#C8452C; margin-bottom:.3rem; }
.hero h1 { font-size: 3rem; margin: 0; line-height: 1; }
.hero p { color: #6B6358; margin: .4rem 0 1.2rem; font-size: 1.05rem; }
.hero b { color: #C8452C; }
code, .trace, .trace * { font-family: 'IBM Plex Mono', monospace !important; font-size: .82rem; }
.step { border-left: 3px solid #C8452C; padding: .15rem .7rem; margin: .35rem 0; background: #FBF8F1; border-radius: 0 6px 6px 0; }
.step.obs { border-left-color: #1F6F6B; }
.step.plan { border-left-color: #1D1B18; }
.step.thought { border-left-color: #B8A98C; color: #6B6358; }
.tag { display:inline-block; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:#6B6358; margin-right:.5rem; }
.src { border:1px solid #D8CDB6; border-radius:8px; padding:.5rem .7rem; margin:.3rem 0; background:#FBF8F1; font-size:.88rem; }
.chip { display:inline-block; padding:.1rem .55rem; border-radius:99px; background:#EBE3D2; font-size:.78rem; margin:.1rem .2rem .1rem 0; }
[data-testid="stSidebar"] { border-right: 1px solid #D8CDB6; }
[data-testid="stChatMessage"] { background: transparent; border-left: 2px solid #D8CDB6; border-radius: 0; padding-left: 1rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border-left-color: #C8452C; }
</style>
""", unsafe_allow_html=True)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return Embedder()


def api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("key_input", "")


def render_event(box, ev):
    t = ev["type"]
    if t == "plan":
        p = ev["plan"]
        subs = "".join(f"<li>{q}</li>" for q in p.get("sub_questions", []))
        box.markdown(f"<div class='trace'><div class='step plan'><span class='tag'>plan · {p.get('complexity')}</span>"
                     f"{p.get('strategy', '')}<ul>{subs}</ul></div></div>", unsafe_allow_html=True)
    elif t == "thought":
        box.markdown(f"<div class='trace'><div class='step thought'><span class='tag'>thought</span>{ev['text']}</div></div>",
                     unsafe_allow_html=True)
    elif t == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in ev["input"].items())
        box.markdown(f"<div class='trace'><div class='step'><span class='tag'>tool</span>{ev['name']}({args})</div></div>",
                     unsafe_allow_html=True)
    elif t == "observation":
        box.markdown(f"<div class='trace'><div class='step obs'><span class='tag'>result</span>{ev['summary']}</div></div>",
                     unsafe_allow_html=True)


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f" · p.{s['page']}" if s.get("page") else ""
            st.markdown(f"<div class='src'><b>[{s['n']}]</b> {s['source']}{page}<br>"
                        f"<span style='color:#6B6358'>{s['text'][:420]}{'…' if len(s['text']) > 420 else ''}</span></div>",
                        unsafe_allow_html=True)


def steps_from(events, cited_ids=None):
    """Turn agent trace events into steps for the terrain map."""
    out = []
    for ev in events:
        if ev["type"] == "plan":
            out.append({"kind": "plan", "text": f"plan · {ev['plan'].get('complexity')} · {ev['plan'].get('strategy', '')}"})
        elif ev["type"] == "observation" and ev.get("viz"):
            v = dict(ev["viz"])
            v["text"] = {"search": f"search “{v.get('q', '')}” → {ev['summary']}"}.get(v["kind"], f"{ev['name']} → {ev['summary']}")
            out.append(v)
    if cited_ids is not None:
        out.append({"kind": "answer", "cited": cited_ids, "text": f"answer · {len(cited_ids)} sources cited"})
    return out


def draw_map(state):
    """Render the terrain. Steps before `animate_from` are drawn instantly; later ones animate."""
    html = build_map_html(kb, state)
    with map_slot.container():
        if hasattr(st, "iframe"):                       # newer Streamlit
            st.iframe(html, height=HEIGHT + 6)
        else:
            components.html(html, height=HEIGHT + 6)
    if state:
        state["animate_from"], state["animate_ghost"] = len(state.get("steps", [])), False


def replay(i):
    m = st.session_state.messages[i]["map"]
    st.session_state.map = {**m, "animate_from": 0, "animate_ghost": True}


# ---------------- state ----------------
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase(get_embedder())
    st.session_state.messages = []
    st.session_state.map = {}
kb: KnowledgeBase = st.session_state.kb

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### Knowledge base")
    files = st.file_uploader("Add documents", type=["pdf", "docx", "csv", "txt", "md"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    if c1.button("Index files", use_container_width=True, disabled=not files):
        with st.spinner("Extracting, chunking, embedding…"):
            for f in files:
                try:
                    st.toast(f"{f.name}: {kb.add_document(f.name, f.getvalue())} chunks")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
    if c2.button("Load samples", use_container_width=True):
        with st.spinner("Indexing sample docs…"):
            for fn in sorted(os.listdir(SAMPLE_DIR)):
                with open(os.path.join(SAMPLE_DIR, fn), "rb") as fh:
                    kb.add_document(fn, fh.read())
    for s in kb.sources():
        st.markdown(f"<span class='chip'>{s} · {kb.count(s)}</span>", unsafe_allow_html=True)
    if kb.chunks and st.button("Clear knowledge base", use_container_width=True):
        st.session_state.kb = KnowledgeBase(get_embedder())
        st.session_state.messages, st.session_state.map = [], {}
        st.rerun()

    st.markdown("### Agent")
    if not api_key():
        st.text_input("Anthropic API key", type="password", key="key_input")
    model = st.text_input("Model", DEFAULT_MODEL)
    top_k = st.slider("Passages per search", 2, 8, 4)
    max_steps = st.slider("Max agent steps", 2, 10, 6)
    compare = st.toggle("Compare with Standard RAG", value=False,
                        help="Also answer with one-shot retrieval so you can see the difference.")
    st.caption(f"Embeddings: {get_embedder().name}")

# ---------------- main ----------------
st.markdown("<div class='hero'><div class='kicker'>AGENTIC RAG · FIELD CHART</div><h1>Pathfinder</h1>"
            "<p>An <b>agentic RAG</b> system. It plans, picks its tools, rewrites weak searches, "
            "and answers only from your documents, with sources.</p></div>", unsafe_allow_html=True)

map_slot = st.empty()
draw_map(st.session_state.map)

if not kb.chunks:
    st.info("Add documents in the sidebar, or press **Load samples** to try it out.")

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and m.get("standard"):
            a, b = st.columns(2)
            a.markdown("**Agentic RAG**"); a.markdown(m["content"])
            b.markdown("**Standard RAG**"); b.markdown(m["standard"])
        else:
            st.markdown(m["content"])
        if m.get("trace"):
            with st.expander(f"Agent trace ({m['calls']} tool calls)"):
                for ev in m["trace"]:
                    render_event(st, ev)
        if m.get("sources"):
            render_sources(m["sources"])
        if m.get("map"):
            st.button("↺ Replay route on the map", key=f"replay{i}", on_click=replay, args=(i,))

prompt = st.chat_input("Ask a question about your documents…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not kb.chunks:
        reply = "The knowledge base is empty. Add documents first."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)
    elif not api_key():
        st.warning("Enter your Anthropic API key in the sidebar.")
    else:
        from anthropic import Anthropic
        agent = AgenticRAG(kb, Anthropic(api_key=api_key()), model, max_steps, top_k)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.chat_message("assistant"):
            status = st.status("Planning…", expanded=True)
            trace, final = [], None
            st.session_state.map = {"question": prompt, "steps": [], "animate_from": 0}
            draw_map(st.session_state.map)
            try:
                for ev in agent.run(prompt, history):
                    if ev["type"] == "answer":
                        final = ev
                        break
                    trace.append(ev)
                    status.update(label={"plan": "Planning…", "thought": "Thinking…", "tool_call": "Using tools…",
                                         "observation": "Reading results…"}[ev["type"]])
                    render_event(status, ev)
                    steps = steps_from(trace)
                    if ev["type"] in ("plan", "observation") and steps:
                        st.session_state.map = {"question": prompt, "steps": steps, "animate_from": len(steps) - 1}
                        draw_map(st.session_state.map)
                steps = steps_from(trace, final["cited_ids"])
                st.session_state.map = {"question": prompt, "steps": steps, "animate_from": len(steps) - 1}
                draw_map(st.session_state.map)
                status.update(label=f"Done · {final['tool_calls']} tool calls", state="complete", expanded=False)
                standard = agent.standard(prompt) if compare else None
                map_state = {"question": prompt, "steps": steps, "ghost": None}
                if standard and standard.get("viz"):
                    map_state["ghost"] = standard["viz"]
                    st.session_state.map = {**map_state, "animate_from": len(steps), "animate_ghost": True}
                    draw_map(st.session_state.map)
                if standard:
                    a, b = st.columns(2)
                    a.markdown("**Agentic RAG**"); a.markdown(final["text"])
                    b.markdown("**Standard RAG**"); b.markdown(standard["text"])
                else:
                    st.markdown(final["text"])
                render_sources(final["sources"])
                st.session_state.messages.append({
                    "role": "assistant", "content": final["text"], "sources": final["sources"], "trace": trace,
                    "calls": final["tool_calls"], "standard": standard["text"] if standard else None,
                    "map": map_state})
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Something went wrong: {e}")
