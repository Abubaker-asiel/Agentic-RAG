"""Extract -> clean -> chunk."""
import io
import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class Chunk:
    id: int
    source: str
    page: int | None
    text: str


def extract(name: str, data: bytes):
    """Return (list of (page, text), DataFrame | None)."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)], None
    if ext == "docx":
        import docx
        doc = docx.Document(io.BytesIO(data))
        return [(None, "\n\n".join(p.text for p in doc.paragraphs))], None
    if ext == "csv":
        df = pd.read_csv(io.BytesIO(data))
        return csv_to_pages(name, df), df
    if ext in ("txt", "md"):
        return [(None, data.decode("utf-8", errors="ignore"))], None
    raise ValueError(f"Unsupported file type: .{ext}")


def csv_to_pages(name: str, df: pd.DataFrame, rows_per_block: int = 8, max_rows: int = 3000):
    """Render a table as searchable text: a schema header plus blocks of rows (one chunk each)."""
    head = f"Table {name}: {len(df)} rows. Columns: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items())
    pages = [(None, head)]
    d = df.head(max_rows)
    for i in range(0, len(d), rows_per_block):
        block = d.iloc[i:i + rows_per_block]
        lines = [" | ".join(f"{c}={v}" for c, v in row.items()) for _, row in block.iterrows()]
        pages.append((None, f"Table {name}, rows {i + 1}-{i + len(block)}:\n" + "\n".join(lines)))
    return pages


def clean(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)                      # re-join hyphenated line breaks
    text = re.sub(r"(?<![\n.!?:])\n(?!\n|[-•*\d#])", " ", text)   # un-wrap soft-wrapped lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int = 900, overlap: int = 150):
    units = []  # (text, separator used when joining to the previous unit)
    for para in (p.strip() for p in re.split(r"\n{2,}", text) if p.strip()):
        if len(para) <= size:
            units.append((para, "\n"))
            continue
        for k, u in enumerate(re.split(r"(?<=[.!?])\s+", para)):
            while len(u) > size:
                units.append((u[:size], " "))
                u = u[size:]
            if u.strip():
                units.append((u.strip(), "\n" if k == 0 else " "))
    chunks, cur = [], ""
    for u, sep in units:
        if len(cur) + len(u) + 1 <= size:
            cur = f"{cur}{sep}{u}" if cur else u
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if cur else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else ""
            cur = f"{tail} {u}".strip()
    if cur:
        chunks.append(cur)
    return chunks
