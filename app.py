
import json
import re
from datetime import datetime
from enum import Enum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from google import genai
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ══════════════════════════════════════════════════════════════════
# 1. CONSTANTS  – single source of truth; never use raw strings
# ══════════════════════════════════════════════════════════════════
class Status(str, Enum):
    NOT_STARTED = "not started"
    IN_PROGRESS = "in progress"
    COMPLETED   = "completed"

SUBJECTS = [
    "History", "Polity", "Economy",
    "Geography", "Science", "Current Affairs",
]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
STATUS_LABELS = {
    Status.NOT_STARTED: "⏳ Not Started",
    Status.IN_PROGRESS: "🔄 In Progress",
    Status.COMPLETED:   "✅ Completed",
}
STATUS_FROM_LABEL = {v: k for k, v in STATUS_LABELS.items()}

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# ══════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════
def normalize_status(value) -> Status:
    """
    Convert any raw string → Status enum.
    Unknown / None / empty → Status.NOT_STARTED.
    """
    raw = str(value).strip().lower() if value is not None else ""
    for s in Status:
        if s.value == raw:
            return s
    return Status.NOT_STARTED


def parse_ai_json(raw: str) -> list:
    """
    Robustly extract a JSON array from an AI response.
    Strategy:
      1. Strip markdown fences.
      2. Try direct json.loads.
      3. Fallback: regex for the first [...] block.
      4. Validate each item has required keys.
    Raises ValueError with a human-readable message on failure.
    """
    cleaned = (
        raw.replace("```json", "")
           .replace("```", "")
           .strip()
    )
    # Try direct parse first (cleanest path)
    for candidate in [cleaned]:
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return _validate_quiz_items(data)
        except json.JSONDecodeError:
            pass

    # Fallback: extract first [...] block
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_quiz_items(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inside [...] block is malformed: {e}")

    raise ValueError("No valid JSON array found in the AI response.")


def _validate_quiz_items(items: list) -> list:
    required = {"question", "optionA", "optionB", "optionC", "optionD", "answer"}
    valid = []
    for i, item in enumerate(items):
        missing = required - set(item.keys())
        if missing:
            st.warning(f"Question {i+1} skipped — missing fields: {missing}")
            continue
        # Normalize answer: strip + upper; default to "A" if invalid
        ans = str(item.get("answer", "A")).strip().upper()
        item["answer"] = ans if ans in ("A", "B", "C", "D") else "A"
        item.setdefault("explanation", "No explanation provided.")
        valid.append(item)
    if not valid:
        raise ValueError("All questions failed validation.")
    return valid


def safe_bar_chart(series: pd.Series, ylabel: str, title: str):
    """Render a bar chart safely — never crashes on empty/zero data."""
    if series.empty or series.sum() == 0:
        st.info(f"No data available yet for: **{title}**")
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    series.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    max_val = int(series.max())
    ax.set_ylim(0, max_val * 1.35 if max_val > 0 else 1)
    for i, v in enumerate(series.values):
        ax.text(i, v + max(max_val * 0.02, 0.1), str(v), ha="center", fontsize=8)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# 3. DATABASE LAYER  – all MongoDB in one place
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_db():
    """
    Single cached MongoDB connection.
    Raises and stops the app if the DB is unreachable.
    """
    try:
        client = MongoClient(
            st.secrets["MONGO_URI"],
            serverSelectionTimeoutMS=5_000,
        )
        client.server_info()
        db = client["kribsy"]
        return {
            "topics":    db["topics"],
            "questions": db["questions"],
            "scores":    db["quiz_scores"],
        }
    except PyMongoError as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.stop()


def db_run(operation, *args, **kwargs):
    """
    Wraps any MongoDB call with error handling.
    Returns the result or None on failure.
    """
    try:
        return operation(*args, **kwargs)
    except PyMongoError as e:
        st.error(f"Database error: {e}")
        return None


# ── Typed DB operations ───────────────────────────────────────────
def db_get_topics() -> list:
    return list(db_run(DB["topics"].find) or [])

def db_get_questions() -> list:
    return list(db_run(DB["questions"].find) or [])

def db_get_scores() -> list:
    return list(db_run(DB["scores"].find) or [])

def db_insert_topic(subject: str, topic: str, difficulty: str) -> bool:
    result = db_run(DB["topics"].insert_one, {
        "Subject":    subject,
        "Topic":      topic,
        "Difficulty": difficulty,
        "Status":     Status.NOT_STARTED.value,
    })
    return result is not None

def db_topic_exists(subject: str, topic: str) -> bool:
    return bool(db_run(DB["topics"].find_one, {"Subject": subject, "Topic": topic}))

def db_update_status(subject: str, topic: str, status: Status) -> bool:
    result = db_run(
        DB["topics"].update_one,
        {"Subject": subject, "Topic": topic},
        {"$set": {"Status": status.value}},
    )
    return result is not None

def db_delete_topic(subject: str, topic: str) -> bool:
    result = db_run(DB["topics"].delete_one, {"Subject": subject, "Topic": topic})
    return result is not None

def db_save_score(payload: dict) -> bool:
    result = db_run(DB["scores"].insert_one, {**payload, "date": datetime.now()})
    return result is not None


# ══════════════════════════════════════════════════════════════════
# 4. DATA LOADERS  (cached DataFrames; cleared after writes)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def load_topics() -> pd.DataFrame:
    data = db_get_topics()

    # If DB empty → return safe structure
    if not data:
        return pd.DataFrame(columns=["Subject", "Topic", "Difficulty", "Status"])

    df = pd.DataFrame(data)

    # Remove Mongo ID safely
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    # 🔥 FORCE REQUIRED COLUMNS (MOST IMPORTANT FIX)
    required_cols = ["Subject", "Topic", "Difficulty", "Status"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = "Unknown" if col != "Status" else Status.NOT_STARTED.value

    # Clean NaN values
    df["Subject"] = df["Subject"].fillna("Unknown")
    df["Topic"] = df["Topic"].fillna("Unknown")
    df["Difficulty"] = df["Difficulty"].fillna("Easy")
    df["Status"] = df["Status"].fillna(Status.NOT_STARTED.value)

    # Normalize status
    df["Status"] = df["Status"].apply(lambda v: normalize_status(v).value)

    return df



@st.cache_data(ttl=30)
def load_questions() -> pd.DataFrame:
    data = db_get_questions()
    if not data:
        return pd.DataFrame(columns=[
            "Subject", "Question",
            "OptionA", "OptionB", "OptionC", "OptionD",
            "Answer", "Explanation",
        ])
    df = pd.DataFrame(data).drop(columns=["_id"], errors="ignore")
    df["Answer"] = (
        df["Answer"].fillna("A").astype(str).str.strip().str.upper()
    )
    df["Explanation"] = df["Explanation"].fillna("No explanation available.")
    return df


def refresh_data():
    """Clear cache and reload — call after any DB write."""
    st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════
# 5. UNIFIED SESSION STATE
# All keys registered once at startup under a single namespace.
# ══════════════════════════════════════════════════════════════════
_SESSION_DEFAULTS = {
    # Dashboard
    "dash": {
        "last_prompt": "",
        "ai_error":    False,
    },
    # AI Quiz
    "aiq": {
        "questions":   [],
        "index":       0,
        "score":       0,
        "score_saved": False,
        "topic":       "",
        "submitted":   False,   # tracks if current answer was submitted
    },
    # PYQ Quiz
    "pyq": {
        "data":        None,
        "index":       0,
        "score":       0,
        "score_saved": False,
        "subject":     "",
        "submitted":   False,
    },
}

def init_session():
    """Initialise all session state groups exactly once."""
    for group, defaults in _SESSION_DEFAULTS.items():
        if group not in st.session_state:
            st.session_state[group] = dict(defaults)
        else:
            # Fill any missing keys added in later versions
            for k, v in defaults.items():
                st.session_state[group].setdefault(k, v)

def ss(group: str) -> dict:
    """Shorthand accessor: ss('aiq')['index']"""
    return st.session_state[group]

def reset_aiq():
    st.session_state["aiq"] = dict(_SESSION_DEFAULTS["aiq"])

def reset_pyq():
    st.session_state["pyq"] = dict(_SESSION_DEFAULTS["pyq"])


# ══════════════════════════════════════════════════════════════════
# 6. GEMINI CLIENT
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_gemini():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def ask_gemini(prompt: str) -> str:
    gemini = get_gemini()
    for model in GEMINI_MODELS:
        try:
            resp = gemini.models.generate_content(model=model, contents=prompt)
            return resp.text
        except Exception as e:
            print(f"[Gemini] {model} failed: {e}")
    return "All models are busy. Try again later."


# ══════════════════════════════════════════════════════════════════
# 7. APP BOOTSTRAP
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TNPSC PrepAI", page_icon="📚", layout="wide")

DB = get_db()          # cached connection
init_session()         # unified state

topics_df    = load_topics()
questions_df = load_questions()


# ══════════════════════════════════════════════════════════════════
# 8. SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Page",
    ["Dashboard", "Subjects", "Study Tracker", "Quiz", "Report"],
)


# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "Dashboard":

    st.title("📚 Kribsy AI – TNPSC Smart Preparation Assistant")
    st.info(
        "AI-powered TNPSC preparation: track topics, generate quizzes, "
        "practice PYQs, and get AI explanations — all in one place."
    )

    # ── Metrics ───────────────────────────────────────────────────
    total     = len(topics_df)
    completed = int((topics_df["Status"] == Status.COMPLETED.value).sum()) if not topics_df.empty else 0
    subjects  = topics_df["Subject"].nunique() if not topics_df.empty else 0
    ratio     = completed / total if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects Studied",  subjects)
    c2.metric("Topics Added",      total)
    c3.metric("Completed Topics",  completed)
    c4.metric("Completion Rate",   f"{round(ratio * 100, 1)}%")
    st.progress(ratio)

    st.divider()

    # ── AI Topic Explainer ────────────────────────────────────────
    st.subheader("🤖 AI Topic Explainer")

    _dash = ss("dash")
    topic_input = st.text_input("Enter a topic to explain", key="dash_topic_input")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("💡 Explain Topic", key="dash_explain_btn"):
            if not topic_input.strip():
                st.warning("Please enter a topic first.")
            else:
                _dash["last_prompt"] = topic_input.strip()
                _dash["ai_error"]    = False

    with col_btn2:
        if _dash["ai_error"]:
            if st.button("🔁 Retry", key="dash_retry_btn"):
                _dash["ai_error"] = False
                # Keep last_prompt so it re-runs automatically below
                st.rerun()

    if _dash["last_prompt"].strip():
        with st.spinner("AI is thinking..."):
            _prompt = f"""
You are a TNPSC Group 1 expert faculty member.
Explain the topic in a TNPSC-focused format.

RULES:
- Simple English, use bullet points, keep it exam-oriented.
- Bold important TNPSC keywords.
- Include Prelims AND Mains-relevant facts.
- No long unnecessary paragraphs.

SECTIONS TO COVER (use each as a heading):
Introduction | Definition | Origin / History | Year / Timeline |
Father / Founder / Important Personalities | Features |
Classification / Types | Advantages / Importance | Disadvantages / Challenges |
Real-Life Scenario | Case Study (India / Tamil Nadu) | Current Affairs Link |
Government Schemes / Acts / Policies | TNPSC Important Facts |
Prelims Quick Revision | Mains Answer Summary (5–8 Lines)

TOPIC: {_dash["last_prompt"]}
"""
            _result = ask_gemini(_prompt)

        if _result and "busy" not in _result.lower():
            st.markdown(_result)
            _dash["last_prompt"] = ""
            _dash["ai_error"]    = False
        else:
            _dash["ai_error"] = True
            st.error("AI is busy or failed. Click Retry above.")


# ══════════════════════════════════════════════════════════════════
# PAGE: SUBJECTS
# ══════════════════════════════════════════════════════════════════
elif page == "Subjects":

    st.header("Subject Progress")

    if topics_df.empty:
        st.warning("No subjects found. Add topics in Study Tracker first.")
    else:
        for sub in sorted(topics_df.get("Subject", pd.Series(["Unknown"])).dropna().unique()):
            sub_df   = topics_df[topics_df["Subject"] == sub]
            total_s  = len(sub_df)
            comp_s   = int((sub_df["Status"] == Status.COMPLETED.value).sum())
            prog_s   = int((sub_df["Status"] == Status.IN_PROGRESS.value).sum())
            not_s    = total_s - comp_s - prog_s
            ratio_s  = (comp_s + 0.5 * prog_s) / total_s if total_s > 0 else 0.0

            with st.expander(
                f"**{sub}** — {round(ratio_s * 100, 1)}% complete", expanded=False
            ):
                st.progress(ratio_s)
                ca, cb, cc = st.columns(3)
                ca.metric("✅ Completed",   comp_s)
                cb.metric("🔄 In Progress", prog_s)
                cc.metric("⏳ Not Started", not_s)

                # Topic-level detail table
                _display = sub_df[["Topic", "Difficulty", "Status"]].copy()
                _display["Status"] = _display["Status"].apply(
                    lambda v: STATUS_LABELS.get(normalize_status(v), v)
                )
                st.dataframe(_display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: STUDY TRACKER
# ══════════════════════════════════════════════════════════════════
elif page == "Study Tracker":

    st.header("Study Tracker")

    # ── Add Topic ─────────────────────────────────────────────────
    st.subheader("➕ Add New Topic")

    add_sub  = st.selectbox("Subject",    SUBJECTS,     key="add_sub_sel")
    add_top  = st.text_input("Topic Name",              key="add_top_inp")
    add_diff = st.selectbox("Difficulty", DIFFICULTIES, key="add_diff_sel")

    if st.button("💾 Save Topic", key="save_topic_btn"):
        if not add_top.strip():
            st.warning("Topic name cannot be empty.")
        elif db_topic_exists(add_sub, add_top.strip()):
            st.warning("This topic already exists for the selected subject.")
        else:
            if db_insert_topic(add_sub, add_top.strip(), add_diff):
                st.success(f"✅ Topic '{add_top.strip()}' saved under {add_sub}.")
                refresh_data()
                st.rerun()
            else:
                st.error("Failed to save topic. Check database connection.")

    # ── Saved Topics Table ────────────────────────────────────────
    st.subheader("📋 Saved Topics")
    if topics_df.empty:
        st.info("No topics yet. Add one above.")
    else:
        _display_df = topics_df.copy()
        _display_df["Status"] = _display_df["Status"].apply(
            lambda v: STATUS_LABELS.get(normalize_status(v), v)
        )
        st.dataframe(_display_df, use_container_width=True, hide_index=True)

    # ── Update Status ─────────────────────────────────────────────
    st.subheader("✏️ Update Topic Status")

    if topics_df.empty:
        st.warning("No topics to update.")
    else:
        upd_sub = st.selectbox(
            "Filter by Subject",
            sorted(topics_df.get("Subject", pd.Series(["Unknown"])).dropna().unique()),
            key="upd_sub_sel",
        )
        _filtered = topics_df[topics_df.get("Subject", "") == upd_sub]
        upd_top   = st.selectbox(
            "Select Topic",
            _filtered["Topic"].unique(),
            key="upd_top_sel",
        )
        upd_label  = st.selectbox(
            "New Status",
            list(STATUS_LABELS.values()),
            key="upd_status_sel",
        )
        upd_status = STATUS_FROM_LABEL[upd_label]

        if st.button("✅ Update Status", key="upd_status_btn"):
            if db_update_status(upd_sub, upd_top, upd_status):
                st.success(f"Status updated to '{upd_label}' for '{upd_top}'.")
                refresh_data()
                st.rerun()
            else:
                st.error("Update failed. Check database connection.")

    # ── Delete Topic ──────────────────────────────────────────────
    st.subheader("🗑 Delete Topic")

    if not topics_df.empty:
        del_sub = st.selectbox(
            "Subject (delete)",
            sorted(topics_df.get("Subject", pd.Series(["Unknown"])).dropna().unique()),
            key="del_sub_sel",
        )
        _del_filtered = topics_df[topics_df["Subject"] == del_sub]
        del_top       = st.selectbox(
            "Topic to Delete",
            _del_filtered["Topic"].unique(),
            key="del_top_sel",
        )
        del_confirm = st.checkbox("I confirm I want to permanently delete this topic", key="del_confirm")

        if st.button("🗑 Delete Topic", key="del_topic_btn"):
            if not del_confirm:
                st.warning("Tick the confirmation checkbox first.")
            elif db_delete_topic(del_sub, del_top):
                st.success(f"Topic '{del_top}' deleted.")
                refresh_data()
                st.rerun()
            else:
                st.error("Delete failed. Check database connection.")


# ══════════════════════════════════════════════════════════════════
# PAGE: QUIZ
# ══════════════════════════════════════════════════════════════════
elif page == "Quiz":

    st.header("TNPSC Quiz")

    # ─────────────────────────────────────────────────────────────
    # SECTION A: AI Quiz
    # State namespace: ss("aiq")
    # ─────────────────────────────────────────────────────────────
    st.subheader("🤖 AI Quiz Generator")

    _aiq = ss("aiq")

    aiq_topic = st.text_input("Quiz Topic", key="aiq_topic_input")
    aiq_num   = st.selectbox("Number of Questions", [5, 10, 15, 20], key="aiq_num_sel")
    aiq_diff  = st.selectbox("Difficulty Level", DIFFICULTIES, key="aiq_diff_sel")

    if st.button("🎯 Generate AI Quiz", key="aiq_generate_btn"):
        if not aiq_topic.strip():
            st.warning("Please enter a topic.")
        else:
            _gen_prompt = f"""
You are a TNPSC Group 1 question setter.
Generate exactly {aiq_num} {aiq_diff}-level questions on: {aiq_topic.strip()}

Return ONLY a valid JSON array — no markdown, no extra text.

Format:
[
  {{
    "question": "...",
    "optionA": "...",
    "optionB": "...",
    "optionC": "...",
    "optionD": "...",
    "answer": "A",
    "explanation": "..."
  }}
]

Rules:
- TNPSC Group 1 exam style
- One correct answer only (A / B / C / D)
- Clear, detailed explanation per question
- Use double quotes; no apostrophes inside values
- Strictly return only the JSON array
"""
            with st.spinner("Generating quiz..."):
                _raw = ask_gemini(_gen_prompt)

            try:
                _questions = parse_ai_json(_raw)
                reset_aiq()
                _aiq = ss("aiq")
                _aiq["questions"] = _questions
                _aiq["topic"]     = aiq_topic.strip()
                st.success(f"✅ {len(_questions)} questions generated!")
            except ValueError as _ve:
                st.error(f"Quiz generation failed: {_ve}")
                with st.expander("Show raw AI response"):
                    st.text(_raw)

    # ── AI Quiz Player ────────────────────────────────────────────
    if _aiq["questions"]:
        st.divider()
        _qs  = _aiq["questions"]
        _idx = _aiq["index"]

        if _idx < len(_qs):
            _q = _qs[_idx]
            st.subheader(f"Question {_idx + 1} / {len(_qs)}")
            st.progress((_idx + 1) / len(_qs))
            st.markdown(f"**{_q['question']}**")

            _opts = {
                "A": _q["optionA"],
                "B": _q["optionB"],
                "C": _q["optionC"],
                "D": _q["optionD"],
            }
            # Key incorporates topic hash + index → unique across reruns
            _radio_key = f"aiq_r_{abs(hash(_aiq['topic']))}_{_idx}"
            _selected  = st.radio(
                "Your answer:", list(_opts.values()), key=_radio_key
            )

            _c1, _c2 = st.columns(2)
            with _c1:
                if st.button("✅ Submit", key=f"aiq_sub_{_idx}") and not _aiq["submitted"]:
                    _correct = _opts[_q["answer"]]
                    if _selected == _correct:
                        st.success("Correct ✅")
                        _aiq["score"] += 1
                    else:
                        st.error(f"Wrong ❌ — Correct: **{_correct}**")
                    st.info(f"💡 {_q['explanation']}")
                    _aiq["submitted"] = True

            with _c2:
                if st.button("Next ▶", key=f"aiq_nxt_{_idx}"):
                    _aiq["index"]     += 1
                    _aiq["submitted"]  = False
                    st.rerun()

        else:
            # ── AI Quiz Results ───────────────────────────────────
            _sc  = _aiq["score"]
            _tot = len(_qs)
            _pct = round(_sc / _tot * 100, 2) if _tot else 0

            st.success(f"🎉 Score: {_sc} / {_tot}  ({_pct}%)")
            if _pct >= 80:   st.success("Excellent TNPSC Preparation 🔥")
            elif _pct >= 60: st.info("Good Progress 👍")
            else:            st.warning("Needs More Practice 📚")

            if not _aiq["score_saved"]:
                db_save_score({
                    "type": "ai_quiz", "topic": _aiq["topic"],
                    "score": _sc, "total": _tot, "percentage": _pct,
                })
                _aiq["score_saved"] = True

            if st.button("🔄 Restart AI Quiz", key="aiq_restart_btn"):
                reset_aiq()
                st.rerun()

    # ─────────────────────────────────────────────────────────────
    # SECTION B: PYQ Quiz
    # State namespace: ss("pyq")
    # Filtered by topics the student has studied (completed / in progress)
    # with optional difficulty filter.
    # ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📜 PYQ Quiz Generator")

    _pyq = ss("pyq")

    if questions_df.empty:
        st.warning("No PYQ questions found. Please add questions to the database.")
    else:
        # Build studied-subjects set (case-insensitive, space-safe)
        _studied = set()
        if not topics_df.empty:
            _studied = set(
                topics_df[
                    topics_df["Status"].isin([
                        Status.COMPLETED.value, Status.IN_PROGRESS.value
                    ])
                ]["Subject"]
                .dropna()
                .str.strip()
                .str.lower()
            )

        _avail_subjects = sorted([
            s for s in questions_df["Subject"].dropna().unique()
            if (not _studied) or s.strip().lower() in _studied
        ])

        if not _avail_subjects:
            st.warning(
                "No studied subjects with PYQs found. "
                "Mark topics as In Progress or Completed first."
            )
        else:
            pyq_sub  = st.selectbox("Subject", _avail_subjects, key="pyq_sub_sel")
            pyq_diff = st.selectbox(
                "Filter by Difficulty (optional)",
                ["All"] + DIFFICULTIES,
                key="pyq_diff_sel",
            )

            # Reset state when subject changes
            if pyq_sub != _pyq["subject"]:
                reset_pyq()
                _pyq = ss("pyq")
                _pyq["subject"] = pyq_sub

            if st.button("▶ Start PYQ Quiz", key="pyq_start_btn"):
                _pool = questions_df[questions_df["Subject"] == pyq_sub].copy()

                # Difficulty filter — match against topics table
                if pyq_diff != "All" and not topics_df.empty:
                    _diff_topics = set(
                        topics_df[
                            (topics_df["Subject"] == pyq_sub) &
                            (topics_df["Difficulty"] == pyq_diff)
                        ]["Topic"].str.strip().str.lower()
                    )
                    if _diff_topics:
                        # Filter questions whose topic text contains a studied topic keyword
                        _mask = _pool["Question"].str.lower().apply(
                            lambda q: any(t in q for t in _diff_topics)
                        )
                        _filtered_pool = _pool[_mask]
                        _pool = _filtered_pool if not _filtered_pool.empty else _pool

                reset_pyq()
                _pyq = ss("pyq")
                _pyq["data"]    = _pool.sample(min(20, len(_pool))).reset_index(drop=True)
                _pyq["subject"] = pyq_sub
                st.rerun()

            # ── PYQ Quiz Player ───────────────────────────────────
            if _pyq["data"] is not None:
                _pqs  = _pyq["data"]
                _pidx = _pyq["index"]

                if _pidx < len(_pqs):
                    _prow = _pqs.iloc[_pidx]
                    st.subheader(f"Question {_pidx + 1} / {len(_pqs)}")
                    st.progress((_pidx + 1) / len(_pqs))
                    st.markdown(f"**{_prow['Question']}**")

                    _popts = [
                        _prow["OptionA"], _prow["OptionB"],
                        _prow["OptionC"], _prow["OptionD"],
                    ]
                    _p_radio_key = f"pyq_r_{abs(hash(pyq_sub))}_{_pidx}"
                    _p_answer    = st.radio("Your answer:", _popts, key=_p_radio_key)

                    _pc1, _pc2 = st.columns(2)
                    with _pc1:
                        if st.button("✅ Submit", key=f"pyq_sub_{_pidx}") and not _pyq["submitted"]:
                            _p_correct = str(_prow.get("Answer", "")).strip()
                            if not _p_correct:
                                st.warning("No answer recorded for this question.")
                            elif _p_answer == _p_correct:
                                st.success("Correct ✅")
                                _pyq["score"] += 1
                            else:
                                st.error(f"Wrong ❌ — Correct: **{_p_correct}**")
                            st.info(f"💡 {_prow.get('Explanation', 'No explanation.')}")
                            _pyq["submitted"] = True

                    with _pc2:
                        if st.button("Next ▶", key=f"pyq_nxt_{_pidx}"):
                            _pyq["index"]     += 1
                            _pyq["submitted"]  = False
                            st.rerun()

                else:
                    # ── PYQ Results ───────────────────────────────
                    _ps  = _pyq["score"]
                    _pt  = len(_pqs)
                    _pp  = round(_ps / _pt * 100, 2) if _pt else 0

                    st.success(f"🎉 Score: {_ps} / {_pt}  ({_pp}%)")
                    if _pp >= 80:   st.success("Excellent TNPSC Preparation 🔥")
                    elif _pp >= 60: st.info("Good Progress 👍")
                    else:           st.warning("Needs More Practice 📚")

                    if not _pyq["score_saved"]:
                        db_save_score({
                            "type": "pyq_quiz", "subject": pyq_sub,
                            "score": _ps, "total": _pt, "percentage": _pp,
                        })
                        _pyq["score_saved"] = True

                    if st.button("🔄 Restart PYQ Quiz", key="pyq_restart_btn"):
                        reset_pyq()
                        st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ══════════════════════════════════════════════════════════════════
elif page == "Report":

    st.header("📊 Performance Report")

    # ── Study Progress ────────────────────────────────────────────
    st.subheader("Study Overview")

    if topics_df.empty:
        st.warning("No topics yet. Add topics in Study Tracker first.")
    else:
        _tot   = len(topics_df)
        _comp  = int((topics_df["Status"] == Status.COMPLETED.value).sum())
        _prog  = int((topics_df["Status"] == Status.IN_PROGRESS.value).sum())
        _ns    = _tot - _comp - _prog
        _ratio = _comp / _tot if _tot else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Topics",    _tot)
        m2.metric("✅ Completed",    _comp)
        m3.metric("🔄 In Progress",  _prog)
        m4.metric("⏳ Not Started",  _ns)
        m5.metric("Completion %",    f"{round(_ratio*100,1)}%")
        st.progress(_ratio)

        # Topics per Subject
        safe_bar_chart(
            topics_df.get("Subject", pd.Series(["Unknown"])).fillna("Unknown").value_counts(),
            ylabel="Topics",
            title="Topics per Subject",
        )

        # Status Distribution
        _status_counts = (
            topics_df["Status"]
            .apply(lambda v: STATUS_LABELS.get(normalize_status(v), v))
            .value_counts()
        )
        safe_bar_chart(_status_counts, ylabel="Count", title="Study Status Distribution")

        # Subject × Status pivot
        st.subheader("Subject × Status Breakdown")
        _pivot = (
            topics_df
            .assign(Status=topics_df["Status"].apply(
                lambda v: STATUS_LABELS.get(normalize_status(v), v)
            ))
            .groupby(["Subject", "Status"])
            .size()
            .unstack(fill_value=0)
        )
        st.dataframe(_pivot, use_container_width=True)

        # Difficulty Distribution
        safe_bar_chart(
            topics_df["Difficulty"].fillna("Unknown").value_counts(),
            ylabel="Topics",
            title="Topics by Difficulty",
        )

        # Weak area detector
        st.subheader("⚠️ Weak Areas (Not Started + Easy topics)")
        _weak = topics_df[
            (topics_df["Status"] == Status.NOT_STARTED.value) &
            (topics_df["Difficulty"] == "Easy")
        ][["Subject", "Topic", "Difficulty"]].reset_index(drop=True)
        if _weak.empty:
            st.success("No easy topics left unstarted — great job!")
        else:
            st.dataframe(_weak, use_container_width=True, hide_index=True)

    # ── Quiz Score History ────────────────────────────────────────
    st.divider()
    st.subheader("🏆 Quiz Score History")

    _scores_raw = db_get_scores()
    if not _scores_raw:
        st.info("No quiz scores recorded yet. Complete a quiz to see results here.")
    else:
        _sdf = pd.DataFrame(_scores_raw).drop(columns=["_id"], errors="ignore")
        _sdf["date"] = pd.to_datetime(_sdf["date"]).dt.strftime("%d %b %Y  %H:%M")

        # Summary stats
        _sa1, _sa2, _sa3 = st.columns(3)
        _sa1.metric("Total Quizzes Taken", len(_sdf))
        _sa2.metric("Average Score %",     f"{round(_sdf['percentage'].mean(), 1)}%")
        _sa3.metric("Best Score %",        f"{round(_sdf['percentage'].max(), 1)}%")

        # Score trend chart
        _sdf_sorted = _sdf.sort_values("date")
        safe_bar_chart(
            _sdf_sorted.reset_index(drop=True)["percentage"].rename("Score %"),
            ylabel="Score %",
            title="Quiz Score Trend (chronological)",
        )

        # Full table
        _cols = [c for c in ["date", "type", "topic", "subject", "score", "total", "percentage"]
                 if c in _sdf.columns]
        st.dataframe(
            _sdf[_cols].sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
