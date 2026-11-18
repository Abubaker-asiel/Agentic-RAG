"""Embeddings + in-memory vector store."""
import re
import zlib

import numpy as np

from .ingest import Chunk, clean, extract, split_text


class Embedder:
    """fastembed (bge-small, ONNX) with a dependency-free hashing fallback."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.name = "hashing-fallback"
        self._model = None
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=model)
            self.name = model
        except Exception:
            pass

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            v = np.array(list(self._model.embed(texts)), dtype="float32")
        else:
            v = np.stack([self._hash(t) for t in texts])
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def _hash(text: str, dim: int = 1024) -> np.ndarray:
        words = re.findall(r"\w+", text.lower())
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        v = np.zeros(dim, dtype="float32")
        for g in grams:
            v[zlib.crc32(g.encode()) % dim] += 1.0
        return np.log1p(v)


class KnowledgeBase:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 1), dtype="float32")
        self.tables = {}   # source name -> DataFrame
        self.last_q = None
        self._lay = None

    # ---- ingestion ----
    def add_document(self, name: str, data: bytes) -> int:
        if name in self.sources():
            self.remove(name)
        pages, df = extract(name, data)
        new = []
        for page, text in pages:
            pieces = [text] if df is not None else split_text(clean(text))
            for piece in pieces:
                if piece.strip():
                    new.append(Chunk(len(self.chunks) + len(new), name, page, piece))
        if not new:
            raise ValueError(f"No text could be extracted from {name}")
        vecs = self.embedder.embed([c.text for c in new])
        self.vectors = vecs if len(self.chunks) == 0 else np.vstack([self.vectors, vecs])
        self.chunks += new
        if df is not None:
            self.tables[name] = df
        return len(new)

    def remove(self, name: str):
        keep = [i for i, c in enumerate(self.chunks) if c.source != name]
        self.chunks = [Chunk(j, self.chunks[i].source, self.chunks[i].page, self.chunks[i].text) for j, i in enumerate(keep)]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, 1), dtype="float32")
        self.tables.pop(name, None)

    # ---- queries ----
    def sources(self) -> list[str]:
        return list(dict.fromkeys(c.source for c in self.chunks))

    def count(self, source: str) -> int:
        return sum(c.source == source for c in self.chunks)

    def search(self, query: str, k: int = 4, source: str | None = None):
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        self.last_q = q
        scores = self.vectors @ q
        out = []
        for i in np.argsort(-scores):
            c = self.chunks[i]
            if source and c.source != source:
                continue
            out.append((c, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def source_text(self, source: str, max_chars: int = 24000) -> str:
        return "\n\n".join(c.text for c in self.chunks if c.source == source)[:max_chars]

    # ---- terrain map: 2-D PCA of the embeddings ----
    ASPECT = 2.6   # canvas width / height

    def layout(self):
        """Return {'points': [[x, y, source_idx], ...], 'sources': [{name, cx, cy, r}]} in unit coordinates."""
        n = len(self.chunks)
        key = (n, id(self.vectors))
        if self._lay and self._lay["key"] == key:
            return self._lay["out"]
        names = self.sources()
        if n == 0:
            self._lay = {"key": key, "out": {"points": [], "sources": []}}
            return self._lay["out"]
        X = self.vectors
        mu = X.mean(0)
        if n >= 3:
            comps = np.linalg.svd(X - mu, full_matrices=False)[2][:2]
        else:
            comps = np.eye(X.shape[1])[:2]
        P = (X - mu) @ comps.T
        lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
        span = np.where(hi - lo < 1e-6, 1.0, hi - lo)
        srt = np.sort(P, axis=0)
        self._proj = (mu, comps, lo, span, srt)
        Q = self._to_unit(P)
        if n <= 1500:                       # nudge overlapping dots apart
            A = np.array([self.ASPECT, 1.0])
            Z = Q * A + np.random.default_rng(0).normal(0, 1e-4, Q.shape)   # break exact ties
            d0 = min(0.03, 0.55 * np.sqrt(self.ASPECT * 0.74 / n))
            for _ in range(30):
                diff = Z[:, None, :] - Z[None, :, :]
                dist = np.maximum(np.linalg.norm(diff, axis=2) + np.eye(n), 1e-6)
                push = np.where((dist < d0)[..., None], diff / dist[..., None] * (d0 - dist)[..., None] * 0.5, 0).sum(1)
                Z = np.clip(Z + push, [0.03 * self.ASPECT, 0.03], [0.97 * self.ASPECT, 0.97])
            Q = Z / A
        idx = {nm: i for i, nm in enumerate(names)}
        pts = [[round(float(q[0]), 4), round(float(q[1]), 4), idx[c.source]] for q, c in zip(Q, self.chunks)]
        srcs = []
        for nm in names:
            sel = np.array([[p[0], p[1]] for p, c in zip(pts, self.chunks) if c.source == nm])
            c = sel.mean(0)
            r = float(np.sqrt(((sel - c) ** 2).sum(1).mean())) if len(sel) > 1 else 0.04
            srcs.append({"name": nm, "cx": round(float(c[0]), 4), "cy": round(float(c[1]), 4), "r": round(max(r, 0.045), 4)})
        self._lay = {"key": key, "out": {"points": pts, "sources": srcs}}
        return self._lay["out"]

    def _to_unit(self, P):
        """Blend of linear position and rank, so dense clusters get room to breathe."""
        mu, comps, lo, span, srt = self._proj
        P = np.atleast_2d(P)
        lin = np.clip((P - lo) / span, 0, 1)
        rank = np.stack([np.searchsorted(srt[:, a], P[:, a]) / len(srt) for a in range(2)], axis=1)
        return (0.35 * lin + 0.65 * rank) * 0.86 + 0.07

    def project(self, vec) -> list[float]:
        self.layout()
        if not self.chunks:
            return [0.5, 0.5]
        mu, comps = self._proj[:2]
        q = self._to_unit((vec - mu) @ comps.T)[0]
        return [round(float(q[0]), 4), round(float(q[1]), 4)]
