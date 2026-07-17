import json
import re
import os
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
# 0. PAGE CONFIG — must be the very first Streamlit command
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TNPSC PrepAI", page_icon="📚", layout="wide")


# ══════════════════════════════════════════════════════════════════
# 1. AUTH — using Streamlit's built-in login (works on Streamlit
#    Cloud without any custom OAuth flow code). Requires secrets:
#
#    [google]
#    client_id = "..."
#    client_secret = "..."
#    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
#    client_kwargs = {scope = "openid email profile"}
# ══════════════════════════════════════════════════════════════════
if not st.user.get("is_logged_in", False):
    st.title("🔐 Login Required")
    st.markdown("Welcome to **Kribsy** — Smart Study Management with AI Powered Learning For TNPSC.")
    st.button("Sign in with Google", on_click=st.login)
    st.stop()

# ── User is logged in ─────────────────────────────────────────────
CURRENT_USER_EMAIL = st.user.email
CURRENT_USER_NAME  = st.user.name
CURRENT_USER_PIC   = getattr(st.user, "picture", None)


# ══════════════════════════════════════════════════════════════════
# 2. CONSTANTS
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
# 3. HELPERS
# ══════════════════════════════════════════════════════════════════
def normalize_status(value) -> Status:
    raw = str(value).strip().lower() if value is not None else ""
    for s in Status:
        if s.value == raw:
            return s
    return Status.NOT_STARTED


def parse_ai_json(raw: str) -> list:
    cleaned = (
        raw.replace("```json", "")
           .replace("```", "")
           .strip()
    )
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return _validate_quiz_items(data)
    except json.JSONDecodeError:
        pass

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
        ans = str(item.get("answer", "A")).strip().upper()
        item["answer"] = ans if ans in ("A", "B", "C", "D") else "A"
        item.setdefault("explanation", "No explanation provided.")
        valid.append(item)
    if not valid:
        raise ValueError("All questions failed validation.")
    return valid


def safe_bar_chart(series: pd.Series, ylabel: str, title: str):
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
# 4. DATABASE LAYER
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_db():
    try:
        client = MongoClient(
            st.secrets["MONGO_URI"],
            serverSelectionTimeoutMS=5_000,
        )
        client.server_info()
        db = client["kribsy"]
        return {
            "users":     db["users"],
            "topics":    db["topics"],
            "questions": db["questions"],
            "scores":    db["quiz_scores"],
        }
    except PyMongoError as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.stop()


def db_run(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except PyMongoError as e:
        st.error(f"Database error: {e}")
        return None


def db_get_topics() -> list:
    return list(db_run(DB["topics"].find, {"user_email": CURRENT_USER_EMAIL}) or [])

def db_get_questions() -> list:
    return list(db_run(DB["questions"].find) or [])

def db_get_scores() -> list:
    return list(db_run(DB["scores"].find, {"user_email": CURRENT_USER_EMAIL}) or [])

def db_insert_topic(subject: str, topic: str, difficulty: str) -> bool:
    result = db_run(DB["topics"].insert_one, {
        "user_email": CURRENT_USER_EMAIL,
        "Subject":    subject,
        "Topic":      topic,
        "Difficulty": difficulty,
        "Status":     Status.NOT_STARTED.value,
    })
    return result is not None

def db_topic_exists(subject: str, topic: str) -> bool:
    return bool(db_run(DB["topics"].find_one, {
        "user_email": CURRENT_USER_EMAIL,
        "Subject": subject,
        "Topic": topic,
    }))

def db_update_status(subject: str, topic: str, status: Status) -> bool:
    result = db_run(
        DB["topics"].update_one,
        {"user_email": CURRENT_USER_EMAIL, "Subject": subject, "Topic": topic},
        {"$set": {"Status": status.value}},
    )
    return result is not None

def db_delete_topic(subject: str, topic: str) -> bool:
    result = db_run(DB["topics"].delete_one, {
        "user_email": CURRENT_USER_EMAIL,
        "Subject": subject,
        "Topic": topic,
    })
    return result is not None

def db_save_score(payload: dict) -> bool:
    result = db_run(DB["scores"].insert_one, {
        **payload,
        "user_email": CURRENT_USER_EMAIL,
        "date": datetime.now(),
    })
    return result is not None


# ══════════════════════════════════════════════════════════════════
# 5. DATA LOADERS
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def load_topics(_user_email: str) -> pd.DataFrame:
    data = db_get_topics()
    if not data:
        return pd.DataFrame(columns=["Subject", "Topic", "Difficulty", "Status"])
    df = pd.DataFrame(data)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    if "user_email" in df.columns:
        df = df.drop(columns=["user_email"])
    required_cols = ["Subject", "Topic", "Difficulty", "Status"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "Unknown" if col != "Status" else Status.NOT_STARTED.value
    df["Subject"]    = df["Subject"].fillna("Unknown")
    df["Topic"]      = df["Topic"].fillna("Unknown")
    df["Difficulty"] = df["Difficulty"].fillna("Easy")
    df["Status"]     = df["Status"].fillna(Status.NOT_STARTED.value)
    df["Status"]     = df["Status"].apply(lambda v: normalize_status(v).value)
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
    df["Answer"]      = df["Answer"].fillna("A").astype(str).str.strip().str.upper()
    df["Explanation"] = df["Explanation"].fillna("No explanation available.")
    return df


def refresh_data():
    st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════
# 6. SESSION STATE
# ══════════════════════════════════════════════════════════════════
_SESSION_DEFAULTS = {
    "dash": {"last_prompt": "", "ai_error": False},
    "aiq":  {"questions": [], "index": 0, "score": 0,
              "score_saved": False, "topic": "", "submitted": False},
    "pyq":  {"data": None, "index": 0, "score": 0,
              "score_saved": False, "subject": "", "submitted": False},
}

def init_session():
    for group, defaults in _SESSION_DEFAULTS.items():
        if group not in st.session_state:
            st.session_state[group] = dict(defaults)
        else:
            for k, v in defaults.items():
                st.session_state[group].setdefault(k, v)

def ss(group: str) -> dict:
    return st.session_state[group]

def reset_aiq():
    st.session_state["aiq"] = dict(_SESSION_DEFAULTS["aiq"])

def reset_pyq():
    st.session_state["pyq"] = dict(_SESSION_DEFAULTS["pyq"])


# ══════════════════════════════════════════════════════════════════
# 7. GEMINI CLIENT
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
# 8. APP BOOTSTRAP
# ══════════════════════════════════════════════════════════════════
DB = get_db()
init_session()

# Upsert user profile into MongoDB on every login
try:
    DB["users"].update_one(
        {"email": CURRENT_USER_EMAIL},
        {
            "$set": {
                "name":       CURRENT_USER_NAME,
                "picture":    CURRENT_USER_PIC,
                "last_login": datetime.now(),
            },
            "$setOnInsert": {"created_at": datetime.now()},
        },
        upsert=True,
    )
except PyMongoError:
    pass

topics_df    = load_topics(CURRENT_USER_EMAIL)
questions_df = load_questions()


# ══════════════════════════════════════════════════════════════════
# 9. SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    if CURRENT_USER_PIC:
        st.image(CURRENT_USER_PIC, width=50)
    st.write(f"**{CURRENT_USER_NAME}**")
    st.caption(CURRENT_USER_EMAIL)
    if st.button("Logout"):
        st.logout()

    st.divider()
    st.title("Navigation")
    page = st.radio(
        "Choose Page",
        ["Dashboard", "Subjects", "Study Tracker", "Quiz", "Report"],
    )


# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "Dashboard":

    st.title("📚 Kribsy-Smart Study Management with AI Powered Learning For TNPSC")
    st.info(
        "AI-powered TNPSC preparation: track topics, generate quizzes, "
        "practice PYQs, and get AI explanations — all in one place."
    )

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
        for sub in sorted(topics_df["Subject"].dropna().unique()):
            sub_df  = topics_df[topics_df["Subject"] == sub]
            total_s = len(sub_df)
            comp_s  = int((sub_df["Status"] == Status.COMPLETED.value).sum())
            prog_s  = int((sub_df["Status"] == Status.IN_PROGRESS.value).sum())
            not_s   = total_s - comp_s - prog_s
            ratio_s = (comp_s + 0.5 * prog_s) / total_s if total_s > 0 else 0.0

            with st.expander(f"**{sub}** — {round(ratio_s * 100, 1)}% complete", expanded=False):
                st.progress(ratio_s)
                ca, cb, cc = st.columns(3)
                ca.metric("✅ Completed",   comp_s)
                cb.metric("🔄 In Progress", prog_s)
                cc.metric("⏳ Not Started", not_s)

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

    st.subheader("📋 Saved Topics")
    if topics_df.empty:
        st.info("No topics yet. Add one above.")
    else:
        _display_df = topics_df.copy()
        _display_df["Status"] = _display_df["Status"].apply(
            lambda v: STATUS_LABELS.get(normalize_status(v), v)
        )
        st.dataframe(_display_df, use_container_width=True, hide_index=True)

        st.subheader("✏️ Update or Delete a Topic")
        sel_sub    = st.selectbox("Subject", sorted(topics_df["Subject"].unique()), key="upd_sub_sel")
        sub_topics = topics_df[topics_df["Subject"] == sel_sub]["Topic"].tolist()
        if sub_topics:
            sel_top          = st.selectbox("Topic", sub_topics, key="upd_top_sel")
            new_status_label = st.selectbox("New Status", list(STATUS_LABELS.values()), key="upd_status_sel")
            cu, cd = st.columns(2)
            with cu:
                if st.button("Update Status", key="upd_status_btn"):
                    new_status = STATUS_FROM_LABEL[new_status_label]
                    if db_update_status(sel_sub, sel_top, new_status):
                        st.success("Status updated.")
                        refresh_data()
                        st.rerun()
            with cd:
                if st.button("🗑️ Delete Topic", key="del_top_btn"):
                    if db_delete_topic(sel_sub, sel_top):
                        st.success("Topic deleted.")
                        refresh_data()
                        st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE: QUIZ
# ══════════════════════════════════════════════════════════════════
elif page == "Quiz":

    st.header("📝 Quiz")
    quiz_mode = st.radio("Quiz Type", ["AI Quiz", "PYQ Quiz"], horizontal=True, key="quiz_mode_radio")

    # ── AI QUIZ ──────────────────────────────────────────────────
    if quiz_mode == "AI Quiz":
        _aiq = ss("aiq")

        if not _aiq["questions"]:
            st.subheader("Generate a new AI quiz")
            gen_topic   = st.text_input("Topic", key="aiq_topic_inp")
            gen_subject = st.selectbox("Subject", SUBJECTS, key="aiq_subject_sel")
            gen_diff    = st.selectbox("Difficulty", DIFFICULTIES, key="aiq_diff_sel")
            gen_count   = st.slider("Number of questions", 5, 20, 10, key="aiq_count_sld")

            if st.button("🎯 Generate Quiz", key="aiq_generate_btn"):
                if not gen_topic.strip():
                    st.warning("Please enter a topic.")
                else:
                    with st.spinner("Generating questions with AI..."):
                        prompt = f"""
You are a TNPSC Group exams question setter.
Generate exactly {gen_count} multiple-choice questions on the topic
"{gen_topic.strip()}" (Subject: {gen_subject}, Difficulty: {gen_diff}).

Respond with ONLY a raw JSON array, no markdown fences, no commentary.
Each item must have EXACTLY these keys:
"question", "optionA", "optionB", "optionC", "optionD", "answer", "explanation"
"answer" must be exactly one of "A", "B", "C", "D".
"""
                        raw = ask_gemini(prompt)
                        try:
                            parsed = parse_ai_json(raw)
                            _aiq["questions"]   = parsed
                            _aiq["index"]       = 0
                            _aiq["score"]       = 0
                            _aiq["score_saved"] = False
                            _aiq["topic"]       = gen_topic.strip()
                            _aiq["submitted"]   = False
                            st.rerun()
                        except ValueError as e:
                            st.error(f"Couldn't generate a valid quiz: {e}")
        else:
            qlist = _aiq["questions"]
            idx   = _aiq["index"]

            if idx < len(qlist):
                q = qlist[idx]
                st.progress(idx / len(qlist))
                st.subheader(f"Question {idx + 1} of {len(qlist)}")
                st.write(f"**{q['question']}**")

                option_map  = {"A": q["optionA"], "B": q["optionB"],
                               "C": q["optionC"], "D": q["optionD"]}
                choice_label = st.radio(
                    "Select your answer",
                    [f"{k}. {v}" for k, v in option_map.items()],
                    key=f"aiq_choice_{idx}", index=None,
                )

                if not _aiq["submitted"]:
                    if st.button("Submit Answer", key=f"aiq_submit_{idx}"):
                        if choice_label is None:
                            st.warning("Please select an answer first.")
                        else:
                            picked = choice_label.split(".")[0]
                            _aiq["submitted"]   = True
                            _aiq["last_correct"] = (picked == q["answer"])
                            if picked == q["answer"]:
                                _aiq["score"] += 1
                            st.rerun()
                else:
                    if _aiq.get("last_correct"):
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Incorrect. Correct answer: {q['answer']}")
                    st.info(f"💡 {q.get('explanation', 'No explanation provided.')}")

                    if st.button("Next ➡️", key=f"aiq_next_{idx}"):
                        _aiq["index"]    += 1
                        _aiq["submitted"] = False
                        st.rerun()
            else:
                st.subheader("🎉 Quiz Complete!")
                st.metric("Your Score", f"{_aiq['score']} / {len(qlist)}")
                if not _aiq["score_saved"]:
                    db_save_score({"mode": "AI Quiz", "topic": _aiq["topic"],
                                   "score": _aiq["score"], "total": len(qlist)})
                    _aiq["score_saved"] = True
                    refresh_data()
                if st.button("🔁 Start a New Quiz", key="aiq_restart_btn"):
                    reset_aiq()
                    st.rerun()

    # ── PYQ QUIZ ─────────────────────────────────────────────────
    else:
        _pyq = ss("pyq")

        if questions_df.empty:
            st.info("No PYQ questions found in the database yet.")
        else:
            if _pyq["data"] is None:
                st.subheader("Start a PYQ quiz")
                pyq_subject = st.selectbox(
                    "Subject", sorted(questions_df["Subject"].dropna().unique()),
                    key="pyq_subject_sel",
                )
                subset    = questions_df[questions_df["Subject"] == pyq_subject]
                max_q     = len(subset)
                pyq_count = st.slider(
                    "Number of questions", 1, max(max_q, 1), min(10, max_q) or 1,
                    key="pyq_count_sld",
                )
                if st.button("▶️ Start Quiz", key="pyq_start_btn"):
                    sample              = subset.sample(n=min(pyq_count, len(subset)))
                    _pyq["data"]        = sample.reset_index(drop=True).to_dict("records")
                    _pyq["index"]       = 0
                    _pyq["score"]       = 0
                    _pyq["score_saved"] = False
                    _pyq["subject"]     = pyq_subject
                    _pyq["submitted"]   = False
                    st.rerun()
            else:
                qlist = _pyq["data"]
                idx   = _pyq["index"]

                if idx < len(qlist):
                    q = qlist[idx]
                    st.progress(idx / len(qlist))
                    st.subheader(f"Question {idx + 1} of {len(qlist)}")
                    st.write(f"**{q.get('Question', '')}**")

                    option_map   = {"A": q.get("OptionA", ""), "B": q.get("OptionB", ""),
                                    "C": q.get("OptionC", ""), "D": q.get("OptionD", "")}
                    choice_label = st.radio(
                        "Select your answer",
                        [f"{k}. {v}" for k, v in option_map.items()],
                        key=f"pyq_choice_{idx}", index=None,
                    )

                    if not _pyq["submitted"]:
                        if st.button("Submit Answer", key=f"pyq_submit_{idx}"):
                            if choice_label is None:
                                st.warning("Please select an answer first.")
                            else:
                                picked      = choice_label.split(".")[0]
                                correct_ans = str(q.get("Answer", "A")).strip().upper()
                                _pyq["submitted"]    = True
                                _pyq["last_correct"] = (picked == correct_ans)
                                if picked == correct_ans:
                                    _pyq["score"] += 1
                                st.rerun()
                    else:
                        correct_ans = str(q.get("Answer", "A")).strip().upper()
                        if _pyq.get("last_correct"):
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. Correct answer: {correct_ans}")
                        st.info(f"💡 {q.get('Explanation', 'No explanation available.')}")

                        if st.button("Next ➡️", key=f"pyq_next_{idx}"):
                            _pyq["index"]    += 1
                            _pyq["submitted"] = False
                            st.rerun()
                else:
                    st.subheader("🎉 Quiz Complete!")
                    st.metric("Your Score", f"{_pyq['score']} / {len(qlist)}")
                    if not _pyq["score_saved"]:
                        db_save_score({"mode": "PYQ Quiz", "topic": _pyq["subject"],
                                       "score": _pyq["score"], "total": len(qlist)})
                        _pyq["score_saved"] = True
                        refresh_data()
                    if st.button("🔁 Start a New PYQ Quiz", key="pyq_restart_btn"):
                        reset_pyq()
                        st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ══════════════════════════════════════════════════════════════════
elif page == "Report":

    st.header("📈 Report")
    scores = db_get_scores()

    if not scores:
        st.info("No quiz attempts yet. Take a quiz to see your report here.")
    else:
        scores_df = pd.DataFrame(scores).drop(columns=["_id", "user_email"], errors="ignore")
        scores_df = scores_df.sort_values("date", ascending=False)
        scores_df["Percentage"] = (scores_df["score"] / scores_df["total"] * 100).round(1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Quizzes Taken",  len(scores_df))
        c2.metric("Average Score",  f"{scores_df['Percentage'].mean():.1f}%")
        c3.metric("Best Score",     f"{scores_df['Percentage'].max():.1f}%")

        st.divider()

        st.subheader("📊 Score by Topic")
        safe_bar_chart(
            scores_df.groupby("topic")["Percentage"].mean().round(1),
            ylabel="Avg %", title="Average Score by Topic"
        )

        st.subheader("📊 Score by Quiz Mode")
        safe_bar_chart(
            scores_df.groupby("mode")["Percentage"].mean().round(1),
            ylabel="Avg %", title="Average Score by Quiz Mode"
        )

        st.divider()
        st.subheader("🕑 Quiz History")
        _display = scores_df[["date", "mode", "topic", "score", "total", "Percentage"]].copy()
        _display["date"] = pd.to_datetime(_display["date"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(_display, use_container_width=True, hide_index=True)
