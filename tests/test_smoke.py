"""Offline smoke test: scripted fake Claude client + hashing embedder."""
import os, sys
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.agent import AgenticRAG
from rag.store import Embedder, KnowledgeBase
from rag.tools import calculate

S = os.path.join(os.path.dirname(__file__), "..", "sample_docs")


def text(t): return NS(type="text", text=t)
def use(i, name, **inp): return NS(type="tool_use", id=i, name=name, input=inp)


class FakeClient:
    def __init__(self, script):
        self.script, self.messages = list(script), NS(create=self.create)
    def create(self, **kw):
        return NS(content=self.script.pop(0), stop_reason="end_turn")


def kb():
    k = KnowledgeBase(Embedder())
    k._model = None
    for fn in os.listdir(S):
        print(fn, k.add_document(fn, open(os.path.join(S, fn), "rb").read()), "chunks")
    return k


def test_calc():
    assert calculate("7500 + 9800") .endswith("17300")
    assert "error" in calculate("__import__('os')").lower()


def test_agent():
    k = kb()
    script = [
        [text('{"complexity":"multi_step","strategy":"look up prices then add","sub_questions":["price ML","price RAG"],"tools":["analyze_data","calculate"]}')],
        [text("Checking the table."), use("a", "analyze_data", source="courses.csv", operation="top", column="price_egp", n=2)],
        [use("b", "search_knowledge_base", query="group discount company enrollment")],
        [use("c", "calculate", expression="(9800 + 9200) * 0.85")],
        [text("The two priciest courses cost 19,000 EGP; with the 15% group discount, 16,150 EGP [1][2].")],
    ]
    agent = AgenticRAG(k, FakeClient(script))
    events = list(agent.run("Cost of the two priciest courses with group discount?"))
    types = [e["type"] for e in events]
    print(types)
    assert types[0] == "plan" and types[-1] == "answer"
    ans = events[-1]
    assert ans["tool_calls"] == 3 and ans["sources"], ans
    obs = [e["summary"] for e in events if e["type"] == "observation"]
    assert "Deep Learning" in obs[0] or "RAG" in obs[0]
    assert "16150" in obs[2]


def test_standard_and_step_limit():
    k = kb()
    agent = AgenticRAG(k, FakeClient([[text("The handbook says refunds are full within 7 days [1].")]]))
    out = agent.standard("refund policy?")
    assert out["sources"]
    loop = [[text("{}")]] + [[use(str(i), "list_sources")] for i in range(2)] + [[text("Best effort answer.")]]
    a2 = AgenticRAG(k, FakeClient(loop), max_steps=2)
    assert list(a2.run("x"))[-1]["text"] == "Best effort answer."


if __name__ == "__main__":
    test_calc(); test_agent(); test_standard_and_step_limit(); print("ALL OK")
