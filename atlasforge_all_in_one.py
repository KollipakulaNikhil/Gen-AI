"""
AtlasForge — single-file Streamlit app
Stack: Streamlit + Firebase Authentication + SQLite + Groq API.

Setup:
1. pip install streamlit python-dotenv pydantic pypdf requests
2. Copy .env.example to .env and fill in credentials.
3. streamlit run app.py

For Streamlit Community Cloud, add the same values to App -> Settings -> Secrets.
Never commit .env or Firebase service-account credentials.
"""
from __future__ import annotations

import html
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

load_dotenv(override=True)

# ----------------------------- Configuration -----------------------------

def setting(name: str, default: str = "") -> str:
    """Read environment variables locally or Streamlit secrets when deployed."""
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return default


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations (user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, research_id TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages (conversation_id, created_at);
CREATE TABLE IF NOT EXISTS research (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL, summary TEXT NOT NULL,
    data TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_user ON research (user_id, updated_at DESC);
"""


@contextmanager
def connect(db_path: str):
    """Open a short-lived SQLite connection; commit on success, always close."""
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def get_db() -> str:
    """Create the SQLite database file and tables; returns its path."""
    db_path = setting("SQLITE_DB_PATH", "atlasforge.db")
    try:
        with connect(db_path) as conn:
            conn.executescript(SCHEMA)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not open SQLite database '{db_path}': {exc}") from exc
    return db_path


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_groq() -> str:
    api_key = setting("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or Streamlit secrets.")
    return api_key


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------- Schema / AI ------------------------------

class ResearchExtraction(BaseModel):
    record_type: str = Field(description="research_paper, research_idea, lecture_notes, or general")
    title: str
    summary: str
    research_questions: list[str] = Field(default_factory=list)
    key_claims: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    expected_outcomes: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


def generate_with_retry(api_key: str, prompt: str, temperature: float, json_mode: bool = False, attempts: int = 4) -> str:
    """Call Groq chat completions, retrying rate-limit/overload responses with exponential backoff."""
    body = {
        "model": setting("GROQ_MODEL", "openai/gpt-oss-120b"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(attempts):
        try:
            response = requests.post(
                GROQ_URL, json=body, timeout=60, headers={"Authorization": f"Bearer {api_key}"}
            )
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Could not reach Groq: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        if response.ok:
            return (response.json()["choices"][0]["message"].get("content") or "").strip()
        if response.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(float(response.headers.get("retry-after", 2 ** attempt)))
            continue
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            detail = response.text
        if response.status_code in (429, 503):
            raise RuntimeError("Groq is busy or rate-limited right now. Please try again in a minute.")
        raise RuntimeError(f"Groq error {response.status_code}: {detail}")


def run_extraction(client, source_text: str) -> ResearchExtraction:
    if not source_text.strip():
        raise ValueError("Please enter text or upload a document with readable text.")
    prompt = f"""You are AtlasForge, a careful research information extractor.
Extract only information supported by the source. Never invent citations, methods, findings, or outcomes.
For research ideas, describe objectives and expected outcomes as proposals, not established facts.
Use empty arrays when a category is absent. Put absent but useful details in missing_information.
Keep claims concise and traceable to the source.

Respond with a single JSON object matching this JSON schema:
{json.dumps(ResearchExtraction.model_json_schema())}

SOURCE TEXT:
{source_text[:24000]}
"""
    text = generate_with_retry(client, prompt, temperature=0.2, json_mode=True)
    if not text:
        raise ValueError("Groq returned an empty response. Please try again.")
    try:
        parsed = ResearchExtraction.model_validate_json(text)
    except ValidationError as exc:
        raise ValueError("Groq returned data that did not match the required research schema. Retry with clearer input.") from exc
    if not parsed.title.strip() or not parsed.summary.strip():
        raise ValueError("The AI output is missing a title or summary. Add more detail and retry.")
    return parsed


def answer_chat(client, prompt: str, recent_messages: list[dict], research_items: list[dict]) -> str:
    context_lines = []
    for item in research_items[:8]:
        context_lines.append(
            f"Research ID: {item.get('id')}\nTitle: {item.get('title')}\n"
            f"Summary: {item.get('summary')}\nClaims: {', '.join(item.get('key_claims', []))}\n"
            f"Keywords: {', '.join(item.get('keywords', []))}"
        )
    history = "\n".join(
        f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
        for message in recent_messages[-12:]
    )
    research_context = "\n\n".join(context_lines) if context_lines else "No saved research records are available."
    full_prompt = f"""You are AtlasForge, a helpful research assistant. Answer clearly and accurately.
Use saved research context when relevant. Distinguish source-supported facts from hypotheses.
Do not fabricate citations. If the saved context does not answer the question, say what is missing.

SAVED RESEARCH CONTEXT:
{research_context}

RECENT CONVERSATION:
{history}

USER MESSAGE:
{prompt}
"""
    text = generate_with_retry(client, full_prompt, temperature=0.4)
    if not text:
        raise ValueError("Groq returned an empty response. Please try again.")
    return text


# ------------------------------ Firebase ---------------------------------

def firebase_auth_request(endpoint: str, payload: dict) -> dict:
    api_key = setting("FIREBASE_WEB_API_KEY") or setting("FIREBASE_API_KEY")
    if not api_key:
        raise RuntimeError("FIREBASE_WEB_API_KEY is missing. Add it to .env or Streamlit secrets.")
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key}"
    response = requests.post(url, json=payload, timeout=25)
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        message = data.get("error", {}).get("message", "Firebase Authentication request failed")
        friendly = {
            "EMAIL_EXISTS": "An account with this email already exists. Sign in instead.",
            "EMAIL_NOT_FOUND": "No account found for this email.",
            "INVALID_PASSWORD": "Incorrect email or password.",
            "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password.",
            "WEAK_PASSWORD": "Use a password with at least 6 characters.",
            "USER_DISABLED": "This account has been disabled.",
            "INVALID_EMAIL": "Enter a valid email address.",
        }
        raise ValueError(friendly.get(message, message.replace("_", " ").title()))
    return data


def sign_in(email: str, password: str) -> dict:
    data = firebase_auth_request("signInWithPassword", {
        "email": email, "password": password, "returnSecureToken": True
    })
    return {"uid": data["localId"], "email": data.get("email", email), "id_token": data["idToken"]}


def sign_up(email: str, password: str) -> dict:
    data = firebase_auth_request("signUp", {
        "email": email, "password": password, "returnSecureToken": True
    })
    return {"uid": data["localId"], "email": data.get("email", email), "id_token": data["idToken"]}


def list_conversations(db, uid: str) -> list[dict]:
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC", (uid,)
        ).fetchall()
    return [dict(row) for row in rows]


def create_conversation(db, uid: str, title: str = "New conversation") -> str:
    cid = uuid.uuid4().hex
    now = now_iso()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (cid, uid, title[:100], now, now),
        )
    return cid


def get_messages(db, uid: str, cid: str) -> list[dict]:
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE user_id = ? AND conversation_id = ? ORDER BY created_at",
            (uid, cid),
        ).fetchall()
    return [{k: v for k, v in dict(row).items() if v is not None} for row in rows]


def save_message(db, uid: str, cid: str, role: str, content: str, research_id: Optional[str] = None) -> None:
    now = now_iso()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO messages (id, user_id, conversation_id, role, content, research_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, uid, cid, role, content, research_id, now),
        )
        if role == "user":
            title = content.strip().splitlines()[0][:80] if content.strip() else "New conversation"
            conn.execute(
                "UPDATE conversations SET updated_at = ?, title = ? WHERE id = ? AND user_id = ?",
                (now, title, cid, uid),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ? AND user_id = ?", (now, cid, uid)
            )


def research_from_row(row: sqlite3.Row) -> dict:
    item = json.loads(row["data"])
    item.update({
        "id": row["id"], "user_id": row["user_id"], "title": row["title"], "summary": row["summary"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    })
    return item


def list_research(db, uid: str, query: str = "") -> list[dict]:
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT * FROM research WHERE user_id = ? ORDER BY updated_at DESC", (uid,)
        ).fetchall()
    items = [research_from_row(row) for row in rows]
    query = query.strip().lower()
    if query:
        items = [item for item in items if query in json.dumps(item, ensure_ascii=False).lower()]
    return items


def save_research(db, uid: str, record: dict) -> dict:
    rid = uuid.uuid4().hex
    now = now_iso()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO research (id, user_id, title, summary, data, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, uid, record["title"], record["summary"], json.dumps(record, ensure_ascii=False), now, now),
        )
    return {**record, "id": rid, "user_id": uid, "created_at": now, "updated_at": now}


def update_research(db, uid: str, research_id: str, title: str, summary: str) -> None:
    with connect(db) as conn:
        cursor = conn.execute(
            "UPDATE research SET title = ?, summary = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title.strip(), summary.strip(), now_iso(), research_id, uid),
        )
    if cursor.rowcount == 0:
        raise ValueError("Record not found.")


def delete_research(db, uid: str, research_id: str) -> None:
    with connect(db) as conn:
        conn.execute("DELETE FROM research WHERE id = ? AND user_id = ?", (research_id, uid))


# ---------------------------------- UI ------------------------------------

st.set_page_config(page_title="AtlasForge | Research AI", page_icon="🧭", layout="wide")

st.markdown("""
<style>
    :root { --accent: #7c8cff; --accent2: #35d0ba; --card: #151b2b; --line: #262f47; --muted: #98a2bd; }
    .stApp {
        background: radial-gradient(1200px 600px at 85% -10%, rgba(124,140,255,.16), transparent 60%),
                    radial-gradient(900px 500px at -10% 110%, rgba(53,208,186,.10), transparent 60%),
                    #0b0f1a;
        color: #eef0f8;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1 { letter-spacing: -1px; font-weight: 800 !important; }
    h1, h2, h3 { color: #f4f6ff; }
    .brand { font-size: 2.1rem; font-weight: 800; letter-spacing: -1px; }
    .brand span { background: linear-gradient(90deg, var(--accent), var(--accent2));
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .login-head { text-align: center; margin-top: 2rem; }
    .login-head .tagline { margin-bottom: 1rem; }
    .tagline { color: var(--muted); margin: -.2rem 0 1.4rem 0; }
    .muted { color: var(--muted); }
    .hero { padding: 3rem 1rem; text-align: center; border: 1px dashed var(--line);
            border-radius: 20px; background: rgba(21,27,43,.6); margin-bottom: 1rem; }
    .hero h2 { font-size: 2rem; letter-spacing: -.5px; margin-bottom: .3rem; }
    [data-testid="stSidebar"] { background: #0f1424; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: #e6e9f5; }
    [data-testid="stSidebar"] a { color: var(--accent2) !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: var(--muted); }
    [data-testid="stBottom"], [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] { background: transparent !important; }
    [data-testid="stChatInput"] > div { background: var(--card) !important; border: 1px solid var(--line); }
    [data-testid="stChatInput"] textarea { color: #eef0f8 !important; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        background: #0f1424 !important; color: #eef0f8 !important; border-radius: 10px; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; }
    [data-testid="stSidebar"] .stButton > button { text-align: left; justify-content: flex-start; }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 12px; border: 1px solid var(--line); background: var(--card);
        color: #eef0f8; transition: all .15s ease; }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        border-color: var(--accent); transform: translateY(-1px); color: #fff; }
    .stFormSubmitButton > button[kind="primary"], .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, var(--accent), #5a6bff); border: none; font-weight: 600; }
    div[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 18px;
        background: var(--card); padding: 1.4rem; }
    div[data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: 16px;
        padding: 14px; background: rgba(21,27,43,.7); margin-bottom: .6rem; }
    div[data-testid="stChatInput"] { border-radius: 16px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
    div[data-testid="stExpander"] { border-radius: 12px; border-color: var(--line); }
    .chip { display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
            font-size: .78rem; color: #cfd6ff; background: rgba(124,140,255,.14);
            border: 1px solid rgba(124,140,255,.35); }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 8px; font-size: .75rem;
             color: #0b0f1a; background: var(--accent2); font-weight: 700; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("auth_user", None)
st.session_state.setdefault("active_conversation", None)
st.session_state.setdefault("page", "Chat")


def open_conversation(cid: str) -> None:
    # Callbacks run before widgets are instantiated, so the radio's key can be set here.
    st.session_state.active_conversation = cid
    st.session_state.page = "Chat"

if not st.session_state.auth_user:
    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        st.markdown(
            '<div class="login-head"><div class="brand">🧭 <span>AtlasForge</span></div>'
            '<div class="tagline">AI Research &amp; Knowledge Navigator<br>'
            'Turn unstructured input into searchable knowledge</div></div>',
            unsafe_allow_html=True,
        )
        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
        with login_tab:
            with st.form("login_form"):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submit = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if login_submit:
                try:
                    st.session_state.auth_user = sign_in(login_email.strip(), login_password)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with signup_tab:
            with st.form("signup_form"):
                signup_email = st.text_input("Email address", key="signup_email")
                signup_password = st.text_input("Password (minimum 6 characters)", type="password", key="signup_password")
                signup_submit = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if signup_submit:
                try:
                    if len(signup_password) < 6:
                        raise ValueError("Password must contain at least 6 characters.")
                    st.session_state.auth_user = sign_up(signup_email.strip(), signup_password)
                    st.success("Account created.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    st.stop()

st.markdown('<div class="brand">🧭 <span>AtlasForge</span></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">AI Research &amp; Knowledge Navigator · Turn unstructured input into searchable knowledge</div>', unsafe_allow_html=True)

try:
    db = get_db()
    groq_key = get_groq()
except Exception as exc:
    st.error(str(exc))
    st.info("Configure GROQ_API_KEY in .env or Streamlit Community Cloud secrets.")
    st.stop()

user = st.session_state.auth_user
uid = user["uid"]

with st.sidebar:
    st.markdown("## 🧭 AtlasForge")
    st.caption(user.get("email", "Signed in"))
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.active_conversation = create_conversation(db, uid)
        st.session_state.page = "Chat"
        st.rerun()
    st.radio("Workspace", ["Chat", "Analyze input", "Research library"], key="page")
    st.divider()
    st.markdown("**Recent conversations**")
    try:
        conversations = list_conversations(db, uid)
        if not conversations:
            st.caption("Your conversations will appear here.")
        for conv in conversations[:15]:
            label = conv.get("title", "New conversation")
            st.button("💬 " + label[:30], key="open_" + conv["id"], use_container_width=True,
                      on_click=open_conversation, args=(conv["id"],))
    except Exception as exc:
        st.warning(f"Could not load conversations: {exc}")
    st.divider()
    if st.button("Sign out", use_container_width=True):
        st.session_state.auth_user = None
        st.session_state.active_conversation = None
        st.rerun()

page = st.session_state.page

if page == "Chat":
    st.markdown("### Research chat")
    if not st.session_state.active_conversation:
        st.markdown('<div class="hero"><h2>Research, understood.</h2><p class="muted">Ask questions about your saved research or analyze new material.</p></div>', unsafe_allow_html=True)
        if st.button("Start a conversation", type="primary"):
            st.session_state.active_conversation = create_conversation(db, uid)
            st.rerun()
    else:
        cid = st.session_state.active_conversation
        messages = get_messages(db, uid, cid)
        for msg in messages:
            with st.chat_message("assistant" if msg.get("role") == "assistant" else "user"):
                st.markdown(msg.get("content", ""))
                if msg.get("research_id"):
                    st.caption(f"Saved research record: {msg['research_id']}")
        prompt = st.chat_input("Ask about your research or enter a research idea…")
        if prompt:
            save_message(db, uid, cid, "user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching your knowledge and thinking…"):
                    try:
                        saved_items = list_research(db, uid)
                        answer = answer_chat(groq_key, prompt, messages + [{"role": "user", "content": prompt}], saved_items)
                        st.markdown(answer)
                        save_message(db, uid, cid, "assistant", answer)
                    except Exception as exc:
                        st.error(f"Could not generate an answer: {exc}")
            st.rerun()

elif page == "Analyze input":
    st.markdown("### Analyze research input")
    st.write("Paste a research paper excerpt, idea, notes, or upload a text-based PDF.")
    with st.form("analyze_form"):
        source_text = st.text_area("Text to analyze", height=220, placeholder="Paste your research idea, paper abstract, or lecture notes…")
        uploaded_file = st.file_uploader("Or upload a PDF", type=["pdf"])
        submitted = st.form_submit_button("Extract structured research", type="primary", use_container_width=True)
    if submitted:
        extracted_text = source_text.strip()
        if uploaded_file:
            try:
                reader = PdfReader(uploaded_file)
                pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                extracted_text = (extracted_text + "\n\n" + pdf_text).strip()
            except Exception as exc:
                st.error(f"Could not read the PDF: {exc}")
                extracted_text = ""
        if not extracted_text:
            st.warning("Add text or upload a PDF with selectable text.")
        else:
            try:
                with st.spinner("Groq is extracting and validating structured data…"):
                    parsed = run_extraction(groq_key, extracted_text)
                    record = parsed.model_dump()
                    saved = save_research(db, uid, record)
                st.success("Validated research was saved to SQLite.")
                st.markdown(f"### {saved['title']}")
                st.write(saved["summary"])
                st.caption(f"Record ID: {saved['id']}")
                for field, label in [
                    ("research_questions", "Research questions"),
                    ("key_claims", "Key claims"),
                    ("methods", "Methods"),
                    ("limitations", "Limitations"),
                    ("keywords", "Keywords"),
                    ("cited_sources", "Cited sources"),
                    ("objectives", "Objectives"),
                    ("expected_outcomes", "Expected outcomes"),
                    ("missing_information", "Missing information"),
                ]:
                    values = saved.get(field, [])
                    if values:
                        with st.expander(f"{label} ({len(values)})", expanded=(field in ["key_claims", "methods"])):
                            for value in values:
                                st.markdown(f"- {value}")
                if not st.session_state.active_conversation:
                    st.session_state.active_conversation = create_conversation(db, uid, saved["title"])
                cid = st.session_state.active_conversation
                save_message(db, uid, cid, "user", f"Analyze this research material: {saved['title']}")
                save_message(db, uid, cid, "assistant", f"Saved structured research record: **{saved['title']}**\n\n{saved['summary']}", saved["id"])
            except Exception as exc:
                st.error(f"Extraction or database save failed: {exc}")
                st.caption("The app only confirms a save after SQLite commits the record.")

elif page == "Research library":
    st.markdown("### Your research library")
    query = st.text_input("Search saved research", placeholder="Search title, claim, method, keyword…")
    try:
        items = list_research(db, uid, query)
    except Exception as exc:
        st.error(f"Could not load research: {exc}")
        items = []
    st.caption(f"{len(items)} saved record(s)")
    if not items:
        st.info("No matching records yet. Analyze text or a PDF to create your first record.")
    for item in items:
        with st.container(border=True):
            st.markdown(
                f'<span class="badge">{html.escape(str(item.get("record_type", "research")).replace("_", " "))}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"#### {item.get('title', 'Untitled research')}")
            st.caption(f"Saved {item.get('created_at', '')[:10]} · ID: {item.get('id', '')}")
            st.write(item.get("summary", ""))
            keywords = item.get("keywords", [])
            if keywords:
                st.markdown(
                    "".join(f'<span class="chip">{html.escape(str(keyword))}</span>' for keyword in keywords),
                    unsafe_allow_html=True,
                )
            with st.expander("View structured data"):
                st.json({key: value for key, value in item.items() if key not in ["user_id"]})
            col1, col2 = st.columns(2)
            with col1:
                with st.form(f"edit_{item['id']}"):
                    new_title = st.text_input("Edit title", value=item.get("title", ""), key=f"title_{item['id']}")
                    new_summary = st.text_area("Edit summary", value=item.get("summary", ""), key=f"summary_{item['id']}")
                    update_clicked = st.form_submit_button("Update record")
                if update_clicked:
                    try:
                        update_research(db, uid, item["id"], new_title, new_summary)
                        st.success("Record updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Update failed: {exc}")
            with col2:
                if st.button("Delete record", key=f"delete_{item['id']}"):
                    try:
                        delete_research(db, uid, item["id"])
                        st.success("Record deleted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

