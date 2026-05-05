"""
Building Code AI — a free RAG app for asking questions about building
regulation PDFs.

- Shared library: PDFs in ./regulations/ are indexed once and available
  to every user of the deployed app.
- Personal uploads: each user can also upload their own PDFs in their
  browser session (they are NOT shared with other users).
- Multiple chats: the sidebar lets you keep several conversations open
  in parallel without losing the loaded knowledge base.
- Web fallback: if the answer isn't in any indexed PDF, the assistant
  says so plainly and offers a best-effort suggestion from a web search.

Stack: Streamlit + Google Gemini (free tier) + pypdf + numpy.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import streamlit as st
from pypdf import PdfReader

import sys
import google.generativeai as genai


def _log(msg: str) -> None:
    print(f"[bca] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "Building Code AI"
APP_TAGLINE = "Ask questions across your loaded building regulations."

REGULATIONS_DIR = Path(__file__).parent / "regulations"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SHARED_INDEX_PATH = CACHE_DIR / "shared_index.pkl"

CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"

CHUNK_TOKENS = 900           # rough chunk size in characters / ~4 = tokens
CHUNK_OVERLAP = 150
TOP_K = 6                    # retrieved chunks per query
MIN_RELEVANCE = 0.40         # cosine score below which we treat the
                             # library as "no answer found"


# ---------------------------------------------------------------------------
# Helpers — secrets, API key, password gate
# ---------------------------------------------------------------------------

def _get_secret(name: str, default: str = "") -> str:
    """Read a secret from st.secrets first, then env var, then default."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def _configure_gemini() -> bool:
    api_key = _get_secret("GEMINI_API_KEY")
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True


def _password_gate() -> bool:
    """Return True if the user is allowed in. Only enforces a gate if
    APP_PASSWORD is configured."""
    password = _get_secret("APP_PASSWORD", "")
    if not password:
        return True
    if st.session_state.get("_authed"):
        return True
    st.markdown(f"## {APP_TITLE}")
    st.caption("This app is shared by invitation. Please enter the access code.")
    entered = st.text_input("Access code", type="password")
    if st.button("Continue", type="primary"):
        if entered == password:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    return False


# ---------------------------------------------------------------------------
# PDF processing & chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    source: str          # filename (or label) the chunk came from
    page: int            # 1-indexed page number
    text: str            # the chunk text
    embedding: np.ndarray | None = None


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_page(text: str) -> list[str]:
    """Split a page's text into overlapping chunks of approx CHUNK_TOKENS chars."""
    text = _clean(text)
    if len(text) <= CHUNK_TOKENS:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_TOKENS, len(text))
        # try to break on sentence/paragraph boundary
        if end < len(text):
            window = text[end - 200 : end]
            for marker in ("\n\n", ". ", "; ", "\n"):
                idx = window.rfind(marker)
                if idx != -1:
                    end = (end - 200) + idx + len(marker)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


def extract_chunks_from_pdf(file_like, source_label: str) -> list[Chunk]:
    """Return a list of Chunk for every page of the PDF."""
    reader = PdfReader(file_like)
    chunks: list[Chunk] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        for piece in _chunk_page(page_text):
            chunks.append(Chunk(source=source_label, page=i, text=piece))
    return chunks


# ---------------------------------------------------------------------------
# Index — local semantic embeddings via fastembed (no API rate limits)
# ---------------------------------------------------------------------------

from fastembed import TextEmbedding

# Small, fast, semantic. ~130MB download on first run; cached to disk after.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embed_cache = {"model": None}


def _get_embedder() -> TextEmbedding:
    if _embed_cache["model"] is None:
        _log(f"Loading embedding model {EMBED_MODEL_NAME} (first time only)…")
        _embed_cache["model"] = TextEmbedding(model_name=EMBED_MODEL_NAME)
        _log("Embedding model loaded.")
    return _embed_cache["model"]


def _embed_texts(texts: list[str]) -> np.ndarray:
    embedder = _get_embedder()
    vectors = list(embedder.embed(texts))
    return np.stack(vectors).astype(np.float32)


@dataclass
class Index:
    chunks: list[Chunk] = field(default_factory=list)
    matrix: np.ndarray | None = None  # (N, D) normalized

    def _normalize(self, m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def add(self, new_chunks: list[Chunk]) -> None:
        if not new_chunks:
            return
        vectors = _embed_texts([c.text for c in new_chunks])
        for c, v in zip(new_chunks, vectors):
            c.embedding = v
        self.chunks.extend(new_chunks)
        m = np.stack([c.embedding for c in self.chunks])
        self.matrix = self._normalize(m)

    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, k: int = 6) -> list[tuple[float, Chunk]]:
        if self.matrix is None or not self.chunks:
            return []
        qv = _embed_texts([query])[0]
        qv = qv / (np.linalg.norm(qv) or 1.0)
        scores = self.matrix @ qv
        top_idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top_idx]


# ---------------------------------------------------------------------------
# Shared library — built once from ./regulations and cached to disk
# ---------------------------------------------------------------------------

def _files_signature(folder: Path) -> str:
    """Hash the names + sizes + mtimes of PDF files in a folder."""
    h = hashlib.sha256()
    if not folder.exists():
        return "empty"
    for p in sorted(folder.glob("*.pdf")):
        st_ = p.stat()
        h.update(p.name.encode())
        h.update(str(st_.st_size).encode())
        h.update(str(int(st_.st_mtime)).encode())
    return h.hexdigest()


def _files_signature_v2(folder: Path) -> str:
    h = hashlib.sha256()
    h.update(EMBED_MODEL_NAME.encode())
    if not folder.exists():
        return h.hexdigest()
    for p_ in sorted(folder.glob("*.pdf")):
        s = p_.stat()
        h.update(p_.name.encode())
        h.update(str(s.st_size).encode())
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def load_shared_index() -> Index:
    """Build (or load from cache) the embedding index over shared PDFs."""
    sig = _files_signature_v2(REGULATIONS_DIR)
    cache_file = CACHE_DIR / f"index_{sig[:12]}.pkl"

    if cache_file.exists():
        try:
            _log(f"Loading cached index from {cache_file.name}")
            with open(cache_file, "rb") as f:
                payload = pickle.load(f)
            idx = Index(chunks=payload["chunks"])
            idx.matrix = payload["matrix"]
            _log(f"Cache hit: {len(idx.chunks)} chunks loaded.")
            return idx
        except Exception as e:
            _log(f"Cache load failed: {e}; rebuilding")

    idx = Index()
    if not REGULATIONS_DIR.exists():
        return idx
    pdfs = sorted(REGULATIONS_DIR.glob("*.pdf"))
    if not pdfs:
        return idx

    progress = st.progress(0.0, text="Reading regulation PDFs…")
    all_chunks: list[Chunk] = []
    for i, pdf_path in enumerate(pdfs):
        _log(f"Reading PDF {i+1}/{len(pdfs)}: {pdf_path.name}")
        progress.progress(
            i / max(len(pdfs), 1),
            text=f"Reading {pdf_path.name}…",
        )
        with open(pdf_path, "rb") as fh:
            file_chunks = extract_chunks_from_pdf(fh, source_label=pdf_path.name)
        all_chunks.extend(file_chunks)
        _log(f"  -> {pdf_path.name}: {len(file_chunks)} chunks")
    progress.progress(0.5, text=f"Embedding {len(all_chunks)} chunks…")
    _log(f"Embedding {len(all_chunks)} chunks…")
    idx.add(all_chunks)
    _log("Index built.")
    progress.empty()

    try:
        with open(cache_file, "wb") as f:
            pickle.dump({"chunks": idx.chunks, "matrix": idx.matrix}, f)
        _log(f"Cached to {cache_file.name}")
    except Exception as e:
        _log(f"Cache save failed: {e}")

    return idx


# ---------------------------------------------------------------------------
# Personal uploads — kept in session state, not shared
# ---------------------------------------------------------------------------

def get_personal_index() -> Index:
    if "personal_index" not in st.session_state:
        st.session_state["personal_index"] = Index()
    return st.session_state["personal_index"]


def ingest_uploaded_file(uploaded) -> int:
    """Index an UploadedFile into the personal index. Returns chunks added."""
    label = f"(your upload) {uploaded.name}"
    chunks = extract_chunks_from_pdf(io.BytesIO(uploaded.getvalue()), label)
    if not chunks:
        return 0
    get_personal_index().add(chunks)
    return len(chunks)


# ---------------------------------------------------------------------------
# Retrieval across both indexes
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = TOP_K) -> list[tuple[float, Chunk]]:
    shared = load_shared_index().search(query, k=k)
    personal = get_personal_index().search(query, k=k)
    merged = sorted(shared + personal, key=lambda x: -x[0])[:k]
    return merged


# ---------------------------------------------------------------------------
# Web fallback (best-effort, free, no API key)
# ---------------------------------------------------------------------------

def web_search_snippets(query: str, max_results: int = 4) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results)) or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are Building Code AI, an assistant that helps architects, engineers "
    "and contractors find answers in building regulation documents.\n\n"
    "RULES:\n"
    "1. ALWAYS prefer the document context provided to you over general knowledge.\n"
    "2. When you use a fact from the documents, cite it inline like "
    "(Source: <filename>, p.<page>). Use the exact filename and page numbers "
    "given in the context blocks.\n"
    "3. If the answer is not in the provided context, say so clearly with the "
    "phrase: 'I could not find this in the loaded documents.' Then, if web "
    "search results are also provided, you may offer them as a best-effort "
    "suggestion — but make it explicit that this is from the public web and "
    "should be verified against the official code that applies to the user.\n"
    "4. Never invent regulation numbers, clause numbers, or page numbers. If "
    "you are unsure, say so.\n"
    "5. Be concise and practical. Use short paragraphs and bullet lists when "
    "they aid readability.\n"
)


def build_context_block(hits: list[tuple[float, Chunk]]) -> str:
    if not hits:
        return "(no document context retrieved)"
    blocks = []
    for score, c in hits:
        blocks.append(
            f"[Source: {c.source}, p.{c.page}, relevance={score:.2f}]\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


def build_web_block(snippets: list[dict]) -> str:
    if not snippets:
        return ""
    lines = ["(web search results — public internet, treat as suggestions)"]
    for s in snippets:
        title = s.get("title") or ""
        body = s.get("body") or ""
        href = s.get("href") or ""
        lines.append(f"- {title}\n  {body}\n  {href}")
    return "\n".join(lines)


def answer_question(
    question: str, history: list[dict]
) -> tuple[str, list[tuple[float, Chunk]], list[dict]]:
    """Run the RAG pipeline. Returns (answer, hits, web_snippets)."""
    hits = retrieve(question, k=TOP_K)
    best_score = hits[0][0] if hits else 0.0
    web_snippets: list[dict] = []
    if best_score < MIN_RELEVANCE:
        web_snippets = web_search_snippets(question)

    context_block = build_context_block(hits)
    web_block = build_web_block(web_snippets)

    user_message = (
        f"USER QUESTION:\n{question}\n\n"
        f"DOCUMENT CONTEXT:\n{context_block}\n\n"
        f"{web_block}".strip()
    )

    # Build chat history for the model
    model = genai.GenerativeModel(
        CHAT_MODEL, system_instruction=SYSTEM_INSTRUCTION
    )
    chat_history = []
    for turn in history:
        chat_history.append(
            {
                "role": "user" if turn["role"] == "user" else "model",
                "parts": [turn["content"]],
            }
        )
    chat = model.start_chat(history=chat_history)
    resp = chat.send_message(user_message)
    return resp.text, hits, web_snippets


# ---------------------------------------------------------------------------
# Chat session management
# ---------------------------------------------------------------------------

def _new_chat() -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "title": "New chat",
        "messages": [],   # list of {"role": "user"|"assistant", "content": str}
        "created": time.time(),
    }


def get_chats() -> list[dict]:
    if "chats" not in st.session_state:
        st.session_state["chats"] = [_new_chat()]
        st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"]
    return st.session_state["chats"]


def get_active_chat() -> dict:
    chats = get_chats()
    aid = st.session_state.get("active_chat_id", chats[0]["id"])
    for c in chats:
        if c["id"] == aid:
            return c
    return chats[0]


def set_active(chat_id: str) -> None:
    st.session_state["active_chat_id"] = chat_id


def delete_chat(chat_id: str) -> None:
    st.session_state["chats"] = [c for c in get_chats() if c["id"] != chat_id]
    if not st.session_state["chats"]:
        st.session_state["chats"] = [_new_chat()]
    st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"]


def auto_title(question: str) -> str:
    q = question.strip().split("\n", 1)[0]
    return (q[:40] + "…") if len(q) > 40 else q


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        st.caption(APP_TAGLINE)

        st.divider()
        if st.button("➕ New chat", use_container_width=True):
            new = _new_chat()
            st.session_state["chats"].append(new)
            set_active(new["id"])
            st.rerun()

        st.markdown("**Your chats**")
        chats = get_chats()
        active_id = st.session_state.get("active_chat_id")
        for c in reversed(chats):  # newest first
            cols = st.columns([0.85, 0.15])
            label = c["title"] if c["title"] else "Untitled"
            is_active = c["id"] == active_id
            with cols[0]:
                if st.button(
                    ("• " if is_active else "  ") + label,
                    key=f"open_{c['id']}",
                    use_container_width=True,
                ):
                    set_active(c["id"])
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"del_{c['id']}", help="Delete chat"):
                    delete_chat(c["id"])
                    st.rerun()

        st.divider()
        with st.expander("📚 Loaded documents", expanded=False):
            shared = load_shared_index().sources()
            personal = get_personal_index().sources()
            st.caption("Shared library (everyone sees these)")
            if shared:
                for s in shared:
                    st.write(f"• {s}")
            else:
                st.write("_(none yet — drop PDFs into the `regulations/` folder)_")
            st.caption("Your uploads (only this session)")
            if personal:
                for s in personal:
                    st.write(f"• {s}")
            else:
                st.write("_(none yet)_")

        with st.expander("⬆️ Add your own PDF", expanded=False):
            uploaded = st.file_uploader(
                "Upload a PDF for this session only",
                type=["pdf"],
                accept_multiple_files=True,
                key="uploader",
            )
            if uploaded:
                added_total = 0
                for f in uploaded:
                    key = f"_ingested_{f.name}_{f.size}"
                    if st.session_state.get(key):
                        continue
                    with st.spinner(f"Indexing {f.name}…"):
                        try:
                            added_total += ingest_uploaded_file(f)
                            st.session_state[key] = True
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed to index {f.name}: {e}")
                if added_total:
                    st.success(f"Indexed {added_total} chunks.")

        st.divider()
        st.caption(
            "Built with Streamlit + Gemini. The assistant cites the file and "
            "page it used. Always verify against the official, current code."
        )


def render_messages(chat: dict) -> None:
    for m in chat["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("hits"):
                with st.expander("Show retrieved excerpts"):
                    for score, ch in m["hits"]:
                        st.markdown(
                            f"**{ch['source']}** — page {ch['page']} "
                            f"_(relevance {score:.2f})_"
                        )
                        st.text(ch["text"])
                        st.divider()
            if m["role"] == "assistant" and m.get("web"):
                with st.expander("Show web search results used"):
                    for s in m["web"]:
                        st.markdown(
                            f"**[{s.get('title','(no title)')}]({s.get('href','')})**"
                        )
                        st.caption(s.get("body", ""))
                        st.divider()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE, page_icon="🏗️", layout="wide"
    )

    if not _password_gate():
        return

    if not _configure_gemini():
        st.error(
            "No Gemini API key configured. Add `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml` (locally) or to the Streamlit Cloud "
            "Secrets editor (in deployment), then reload."
        )
        st.stop()

    # Render the title FIRST so the user sees something while indexing.
    st.markdown(f"## 🏗️ {APP_TITLE}")
    st.caption(APP_TAGLINE)

    # Pre-warm the shared library index in the main panel so progress is visible.
    if "shared_index_ready" not in st.session_state:
        _log("Pre-warming shared index from main()")
        with st.status(
            "Indexing your regulation PDFs for the first time… "
            "(can take 10+ minutes for large documents — only happens once)",
            expanded=True,
        ) as status:
            try:
                load_shared_index()
                status.update(
                    label="Indexing complete.", state="complete", expanded=False
                )
                st.session_state["shared_index_ready"] = True
            except Exception as e:
                _log(f"Indexing failed: {e}")
                status.update(
                    label=f"Indexing failed: {e}", state="error", expanded=True,
                )
                st.error(
                    f"The PDF indexing failed. You can still chat — answers "
                    f"will fall back to web search.\n\nError: `{e}`"
                )
                st.session_state["shared_index_ready"] = True  # don't retry

    render_sidebar()

    chat = get_active_chat()

    render_messages(chat)

    user_input = st.chat_input("Ask a question about the loaded regulations…")
    if user_input:
        chat["messages"].append({"role": "user", "content": user_input})
        if chat["title"] in ("New chat", "Untitled"):
            chat["title"] = auto_title(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Searching the loaded documents…_")
            try:
                # Build history of all turns BEFORE this one
                history = chat["messages"][:-1]
                answer, hits, web = answer_question(user_input, history)
            except Exception as e:  # noqa: BLE001
                answer = f"Sorry, something went wrong: `{e}`"
                hits, web = [], []
            placeholder.markdown(answer)

            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "hits": [
                        (s, {"source": c.source, "page": c.page, "text": c.text})
                        for s, c in hits
                    ],
                    "web": web,
                }
            )
        st.rerun()


if __name__ == "__main__":
    main()
