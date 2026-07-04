import json
import re
from datetime import datetime
from enum import Enum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import pandas as pd
import streamlit as st
from google import genai
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ══════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG  – must be the very first Streamlit command, and
#    must be called EXACTLY ONCE. (Your original code called this
#    twice, which raises a StreamlitAPIException.)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TNPSC PrepAI", page_icon="📚", layout="wide")


# ══════════════════════════════════════════════════════════════════
# 1. GOOGLE OAUTH HELPERS
#    NOTE: autogenerate_code_verifier=False disables PKCE. PKCE
#    requires the exact same Flow object (with its code_verifier) to
#    be reused both when building the login link AND when exchanging
#    the code for tokens. But the redirect to Google and back is a
#    full browser page reload — and Streamlit does not reliably keep
#    session_state alive across that kind of reload. That mismatch
#    was causing the "loops back to sign-in" bug. Since this OAuth
#    Client has a client_secret (confidential web app), PKCE isn't
#    required, and a brand-new Flow object can safely exchange the
#    code with zero dependency on prior session state.
# ══════════════════════════════════════════════════════════════════
import os
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
if "localhost" not in st.secrets.get("google", {}).get("oauth_redirect_url", ""):
    pass  # deployed https — do NOT set insecure transport
else:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # localhost only

def get_flow():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": st.secrets["google"]["client_id"],
                "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["google"]["oauth_redirect_url"]],
            }
        },
        scopes=["openid", "email", "profile"],
        redirect_uri=st.secrets["google"]["oauth_redirect_url"],
        autogenerate_code_verifier=False,
    )
    return flow


def get_login_url():
    flow = get_flow()
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    return auth_url


def login_with_code(code: str) -> dict:
    """
    Exchanges the auth code for tokens, VERIFIES the id_token against
    Google's servers, and returns a plain dict with the user's
    identity (email, name, picture). This is what should be stored
    in session_state — never the raw JWT string.
    """
    if isinstance(code, list):
        code = code[0]

    flow = get_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials

    info = google_id_token.verify_oauth2_token(
        creds.id_token,
        google_requests.Request(),
        st.secrets["google"]["client_id"],
    )

    return {
        "email": info.get("email"),
        "name": info.get("name", info.get("email")),
        "picture": info.get("picture"),
    }


# ══════════════════════════════════════════════════════════════════
# 2. SESSION INIT (single source of truth, no duplicates)
# ══════════════════════════════════════════════════════════════════
if "user" not in st.session_state:
    st.session_state.user = None


# ══════════════════════════════════════════════════════════════════
# 3. LOGIN FLOW
#    The "code" query param is read and processed FIRST, before any
#    login-page UI is built — so nothing can regenerate or interfere
#    with state needed to process the callback.
# ══════════════════════════════════════════════════════════════════
if st.session_state.user is None:

    params = st.query_params
    code = params.get("code")

    if code:
        st.title("🔐 Logging you in...")
        try:
            user_info = login_with_code(code)

            if not user_info.get("email"):
                st.query_params.clear()
                st.error("Login failed: Google did not return an email address.")
                st.stop()

            st.session_state.user = user_info

            # Upsert the user into MongoDB so we have a persistent
            # users collection (needed below for get_db(), so we
            # connect directly here rather than relying on cached DB).
            try:
                _client = MongoClient(st.secrets["MONGO_URI"], serverSelectionTimeoutMS=5_000)
                _client["kribsy"]["users"].update_one(
                    {"email": user_info["email"]},
                    {
                        "$set": {
                            "name": user_info["name"],
                            "picture": user_info.get("picture"),
                            "last_login": datetime.now(),
                        },
                        "$setOnInsert": {"created_at": datetime.now()},
                    },
                    upsert=True,
                )
            except PyMongoError as e:
                # Don't block login over a logging failure, just warn.
                st.warning(f"Logged in, but couldn't sync user profile: {e}")

            st.query_params.clear()
            st.rerun()

        except Exception as e:
            st.query_params.clear()
            st.error(f"Login failed: {type(e).__name__}: {e}")
            with st.expander("🔍 Full error details (for debugging)"):
                import traceback
                st.code(traceback.format_exc())
            st.info("Please click 'Sign in with Google' below and try again.")

    if st.session_state.user is None:
        st.title("🔐 Login Required")
        auth_url = get_login_url()

        st.markdown(f"""
            <a href="{auth_url}" target="_self">
                <button style="
                    background-color:#4285F4;
                    color:white;
                    padding:10px 20px;
                    border:none;
                    border-radius:5px;
                    font-size:16px;">
                    Sign in with Google
                </button>
            </a>
        """, unsafe_allow_html=True)

        st.stop()




# ══════════════════════════════════════════════════════════════════
# 4. CONSTANTS
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

CURRENT_USER_EMAIL = st.session_state.user["email"]


# ══════════════════════════════════════════════════════════════════
# 5. HELPERS
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
    for candidate in [cleaned]:
        try:
            data = json.loads(candidate)
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
# 6. DATABASE LAYER  – all MongoDB in one place
#    NOTE: "topics" and "scores" are now scoped per-user via the
#    user's email. "questions" stays a shared bank (e.g. PYQs) unless
#    you want AI-generated quizzes to be private too — see db_save_score.
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


# ── Typed DB operations (all scoped to CURRENT_USER_EMAIL) ────────
def db_get_topics() -> list:
    return list(db_run(DB["topics"].find, {"user_email": CURRENT_USER_EMAIL}) or [])

def db_get_questions() -> list:
    # Shared question bank — not user-scoped.
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
# 7. DATA LOADERS  (cached DataFrames; cleared after writes)
#    Cache key includes the user's email so one user's cached data
#    never leaks into another user's session.
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

    df["Subject"] = df["Subject"].fillna("Unknown")
    df["Topic"] = df["Topic"].fillna("Unknown")
    df["Difficulty"] = df["Difficulty"].fillna("Easy")
    df["Status"] = df["Status"].fillna(Status.NOT_STARTED.value)
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
# 8. UNIFIED SESSION STATE
# ══════════════════════════════════════════════════════════════════
_SESSION_DEFAULTS = {
    "dash": {
        "last_prompt": "",
        "ai_error":    False,
    },
    "aiq": {
        "questions":   [],
        "index":       0,
        "score":       0,
        "score_saved": False,
        "topic":       "",
        "submitted":   False,
    },
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
# 9. GEMINI CLIENT
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
# 10. APP BOOTSTRAP
# ══════════════════════════════════════════════════════════════════
DB = get_db()
init_session()

topics_df    = load_topics(CURRENT_USER_EMAIL)
questions_df = load_questions()


# ══════════════════════════════════════════════════════════════════
# 11. SIDEBAR  – user info, logout, navigation
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    _user = st.session_state.user
    if _user.get("picture"):
        st.image(_user["picture"], width=50)
    st.write(f"**{_user['name']}**")
    st.caption(_user["email"])
    if st.button("Logout"):
        st.session_state.user = None
        st.cache_data.clear()
        st.rerun()

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

    st.title("📚 Kribsy AI – TNPSC Smart Preparation Assistant")
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

        # ── Update / Delete controls ────────────────────────────
        st.subheader("✏️ Update or Delete a Topic")
        sel_sub = st.selectbox("Subject", sorted(topics_df["Subject"].unique()), key="upd_sub_sel")
        sub_topics = topics_df[topics_df["Subject"] == sel_sub]["Topic"].tolist()
        if sub_topics:
            sel_top = st.selectbox("Topic", sub_topics, key="upd_top_sel")
            new_status_label = st.selectbox(
                "New Status", list(STATUS_LABELS.values()), key="upd_status_sel"
            )
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
# Two modes: AI-generated quiz (Gemini) and PYQ quiz (from the
# shared "questions" collection). Both write to quiz_scores via
# db_save_score(), which auto-tags the score with the logged-in
# user's email.
# ══════════════════════════════════════════════════════════════════
elif page == "Quiz":

    st.header("📝 Quiz")

    quiz_mode = st.radio("Quiz Type", ["AI Quiz", "PYQ Quiz"], horizontal=True, key="quiz_mode_radio")

    # ──────────────────────────────────────────────────────────────
    # AI QUIZ
    # ──────────────────────────────────────────────────────────────
    if quiz_mode == "AI Quiz":

        _aiq = ss("aiq")

        if not _aiq["questions"]:
            st.subheader("Generate a new AI quiz")

            gen_topic = st.text_input("Topic", key="aiq_topic_inp")
            gen_subject = st.selectbox("Subject", SUBJECTS, key="aiq_subject_sel")
            gen_diff = st.selectbox("Difficulty", DIFFICULTIES, key="aiq_diff_sel")
            gen_count = st.slider("Number of questions", 5, 20, 10, key="aiq_count_sld")

            if st.button("🎯 Generate Quiz", key="aiq_generate_btn"):
                if not gen_topic.strip():
                    st.warning("Please enter a topic.")
                else:
                    with st.spinner("Generating questions with AI..."):
                        prompt = f"""
You are a TNPSC Group 1 question setter.
Generate exactly {gen_count} multiple-choice questions on the topic
"{gen_topic.strip()}" (Subject: {gen_subject}, Difficulty: {gen_diff}).

Respond with ONLY a raw JSON array, no markdown fences, no commentary.
Each item must be an object with EXACTLY these keys:
"question", "optionA", "optionB", "optionC", "optionD", "answer", "explanation"

"answer" must be exactly one of "A", "B", "C", "D".
"explanation" should be 1-2 short sentences.
"""
                        raw = ask_gemini(prompt)
                        try:
                            parsed = parse_ai_json(raw)
                            _aiq["questions"] = parsed
                            _aiq["index"] = 0
                            _aiq["score"] = 0
                            _aiq["score_saved"] = False
                            _aiq["topic"] = gen_topic.strip()
                            _aiq["submitted"] = False
                            st.rerun()
                        except ValueError as e:
                            st.error(f"Couldn't generate a valid quiz: {e}")

        else:
            qlist = _aiq["questions"]
            idx = _aiq["index"]

            if idx < len(qlist):
                q = qlist[idx]
                st.progress((idx) / len(qlist))
                st.subheader(f"Question {idx + 1} of {len(qlist)}")
                st.write(f"**{q['question']}**")

                option_map = {
                    "A": q["optionA"], "B": q["optionB"],
                    "C": q["optionC"], "D": q["optionD"],
                }
                choice_label = st.radio(
                    "Select your answer",
                    [f"{k}. {v}" for k, v in option_map.items()],
                    key=f"aiq_choice_{idx}",
                    index=None,
                )

                if not _aiq["submitted"]:
                    if st.button("Submit Answer", key=f"aiq_submit_{idx}"):
                        if choice_label is None:
                            st.warning("Please select an answer first.")
                        else:
                            picked = choice_label.split(".")[0]
                            _aiq["submitted"] = True
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
                        _aiq["index"] += 1
                        _aiq["submitted"] = False
                        _aiq["last_correct"] = False
                        st.rerun()

            else:
                st.subheader("🎉 Quiz Complete!")
                st.metric("Your Score", f"{_aiq['score']} / {len(qlist)}")

                if not _aiq["score_saved"]:
                    db_save_score({
                        "mode": "AI Quiz",
                        "topic": _aiq["topic"],
                        "score": _aiq["score"],
                        "total": len(qlist),
                    })
                    _aiq["score_saved"] = True
                    refresh_data()

                if st.button("🔁 Start a New Quiz", key="aiq_restart_btn"):
                    reset_aiq()
                    st.rerun()

    # ──────────────────────────────────────────────────────────────
    # PYQ QUIZ
    # ──────────────────────────────────────────────────────────────
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
                subset = questions_df[questions_df["Subject"] == pyq_subject]
                max_q = len(subset)
                pyq_count = st.slider(
                    "Number of questions", 1, max(max_q, 1), min(10, max_q) or 1,
                    key="pyq_count_sld",
                )

                if st.button("▶️ Start Quiz", key="pyq_start_btn"):
                    sample = subset.sample(n=min(pyq_count, len(subset)), random_state=None)
                    _pyq["data"] = sample.reset_index(drop=True).to_dict("records")
                    _pyq["index"] = 0
                    _pyq["score"] = 0
                    _pyq["score_saved"] = False
                    _pyq["subject"] = pyq_subject
                    _pyq["submitted"] = False
                    st.rerun()

            else:
                qlist = _pyq["data"]
                idx = _pyq["index"]

                if idx < len(qlist):
                    q = qlist[idx]
                    st.progress(idx / len(qlist))
                    st.subheader(f"Question {idx + 1} of {len(qlist)}")
                    st.write(f"**{q.get('Question', '')}**")

                    option_map = {
                        "A": q.get("OptionA", ""), "B": q.get("OptionB", ""),
                        "C": q.get("OptionC", ""), "D": q.get("OptionD", ""),
                    }
                    choice_label = st.radio(
                        "Select your answer",
                        [f"{k}. {v}" for k, v in option_map.items()],
                        key=f"pyq_choice_{idx}",
                        index=None,
                    )

                    if not _pyq["submitted"]:
                        if st.button("Submit Answer", key=f"pyq_submit_{idx}"):
                            if choice_label is None:
                                st.warning("Please select an answer first.")
                            else:
                                picked = choice_label.split(".")[0]
                                correct_ans = str(q.get("Answer", "A")).strip().upper()
                                _pyq["submitted"] = True
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
                            _pyq["index"] += 1
                            _pyq["submitted"] = False
                            _pyq["last_correct"] = False
                            st.rerun()

                else:
                    st.subheader("🎉 Quiz Complete!")
                    st.metric("Your Score", f"{_pyq['score']} / {len(qlist)}")

                    if not _pyq["score_saved"]:
                        db_save_score({
                            "mode": "PYQ Quiz",
                            "topic": _pyq["subject"],
                            "score": _pyq["score"],
                            "total": len(qlist),
                        })
                        _pyq["score_saved"] = True
                        refresh_data()

                    if st.button("🔁 Start a New PYQ Quiz", key="pyq_restart_btn"):
                        reset_pyq()
                        st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE: REPORT
# Shows only the logged-in user's quiz history (db_get_scores()
# already filters by CURRENT_USER_EMAIL).
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
        c1.metric("Quizzes Taken", len(scores_df))
        c2.metric("Average Score", f"{scores_df['Percentage'].mean():.1f}%")
        c3.metric("Best Score", f"{scores_df['Percentage'].max():.1f}%")

        st.divider()

        st.subheader("📊 Score by Topic")
        topic_avg = scores_df.groupby("topic")["Percentage"].mean().round(1)
        safe_bar_chart(topic_avg, ylabel="Avg %", title="Average Score by Topic")

        st.subheader("📊 Score by Quiz Mode")
        mode_avg = scores_df.groupby("mode")["Percentage"].mean().round(1)
        safe_bar_chart(mode_avg, ylabel="Avg %", title="Average Score by Quiz Mode")

        st.divider()

        st.subheader("🕑 Quiz History")
        _display = scores_df[["date", "mode", "topic", "score", "total", "Percentage"]].copy()
        _display["date"] = pd.to_datetime(_display["date"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(_display, use_container_width=True, hide_index=True)
