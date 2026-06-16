
import streamlit as st
import pandas as pd
import os
import random
from google import genai
from pymongo import MongoClient

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite"
]

MONGO_URI = st.secrets["MONGO_URI"]

mongo_client = MongoClient(MONGO_URI)

db = mongo_client["kribsy"]

topics_collection = db["topics"]
def ask_gemini(prompt):
    for m in MODELS:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt
            )
            return response.text
        except Exception:
            continue

    return "All models are busy. Try again later."

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

file_path = "data/topics.csv"

if os.path.exists(file_path):
    try:
        topics_df = pd.read_csv(file_path, sep="|")

        topics_df.columns = topics_df.columns.str.strip()

        required_cols = ["Subject", "Topic", "Difficulty", "Status"]

        for col in required_cols:
            if col not in topics_df.columns:
                topics_df[col] = ""

        topics_df = topics_df[required_cols]

        topics_df = topics_df.dropna(how="all")

        topics_df["Subject"] = topics_df["Subject"].astype(str).str.strip()
        topics_df["Topic"] = topics_df["Topic"].astype(str).str.strip()
        topics_df["Difficulty"] = topics_df["Difficulty"].astype(str).str.strip()

        topics_df["Status"] = (
            topics_df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(",", "")
        )

    except Exception as e:
        st.error(f"CSV Load Error: {e}")
        topics_df = pd.DataFrame(columns=["Subject", "Topic", "Difficulty", "Status"])

else:
    topics_df = pd.DataFrame(columns=["Subject", "Topic", "Difficulty", "Status"])

if os.path.exists("data/questions.csv"):
    questions_df = pd.read_csv("data/questions.csv", sep="|")
else:
    questions_df = pd.DataFrame(
        columns=[
            "Subject",
            "Question",
            "OptionA",
            "OptionB",
            "OptionC",
            "OptionD",
            "Answer",
            "Explanation",
        ]
    )

st.set_page_config(
    page_title="TNPSC PrepAI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 kribsy AI-TNPSC Smart Preparation Assistant")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    [
        "Dashboard",
        "Subjects",
        "Study Tracker",
        "Quiz",
        "Report"
    ]
)

if page == "Dashboard":

    st.header("Kribsy AI Dashboard")

    st.info("""
    Kribsy AI is an AI-powered TNPSC preparation assistant
    that helps students track topics, generate quizzes,
    practice similar PYQs, and learn concepts using AI.
    """)

    total_topics = len(topics_df)
    Completed_topics = len(
        topics_df[
            topics_df["Status"].astype(str).str.strip().str.lower() == "Completed"
            ]
    )
    subjects_studied = topics_df["Subject"].nunique() if not topics_df.empty else 0

    if total_topics == 0:
        progress = 0
    else:
        progress = Completed_topics / total_topics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Subjects Studied",
        subjects_studied
    )

    col2.metric(
        "Topics Studied",
        total_topics
    )

    col3.metric(
        "Completed Topics",
        Completed_topics
    )

    col4.metric(
        "completion Rate",
        f"{round(progress * 100, 1)}%"
    )


    st.progress(progress)
    st.write(f"Progress: {round(progress * 100, 2)}%")

    st.divider()

    st.subheader("Kribsy AI")

    topic = st.text_input("Enter topic to explain", key="topic_input_main")

    # initialize memory once

    if "ai_error" not in st.session_state:
        st.session_state.ai_error = False

    # Step 1: Button stores input
    if st.button("Explain topic"):
        if topic.strip() == "":
            st.warning("Please enter a topic first")
        else:
            st.session_state.last_prompt = topic
            st.session_state.ai_error = False

    # Step 2: AI execution
    if st.session_state.last_prompt:

        try:
            with st.spinner("AI is thinking..."):

                # 🔥 YOUR EXACT PROMPT (UNCHANGED)
                prompt = f"""
        You are a TNPSC Group 1 expert faculty member.

Explain the topic in a TNPSC-focused format.

RULES:

- Use simple English.
- Use bullet points wherever possible.
- Keep each section precise and exam-oriented.
- Highlight important TNPSC keywords in bold.
- Include facts useful for Prelims and Mains.
- Add current relevance wherever applicable.
- Do NOT give unnecessary long paragraphs.

FORMAT:

Introduction

- Brief introduction of the topic.

Definition

- Clear definition.

Origin / History

- How and why it originated.

Year / Timeline

- Important years and chronology.

Father / Founder / Important Personalities

- Key persons associated with the topic.

Features

- Main characteristics.

Classification / Types

- Different types/categories (if applicable).

Advantages / Importance

- Key benefits and significance.

Disadvantages / Challenges

- Limitations and issues.

Real-Life Scenario

- Practical application in daily life.

Case Study (India / Tamil Nadu)

- One relevant example.

Current Affairs Link

- Recent developments related to the topic.

Government Schemes / Acts / Policies

- Related schemes, laws, commissions or programmes.

TNPSC Important Facts

- Important one-line facts.
- Dates.
- Committees.
- Constitutional Articles.
- Reports.
- Statistics.

Prelims Quick Revision

- 5–10 important points for objective exams.

Mains Answer Summary (5–8 Lines)

- A ready-to-write TNPSC mains answer conclusion.

TOPIC:
{st.session_state.last_prompt}
        """

                result = ask_gemini(prompt)

                if result and "busy" not in result.lower():
                    st.markdown(result)
                    st.session_state.last_prompt = ""
                    st.session_state.ai_error = False
                else:
                    st.session_state.ai_error = True
                    st.error("AI is busy or failed. Please retry.")

        except Exception as e:
            st.session_state.ai_error = True
            st.error(f"Gemini Error: {e}")

    # Step 3: Retry button (always visible when error happens)
    if st.session_state.ai_error:
        if st.button("🔁 Retry"):
            st.rerun()

elif page == "Subjects":

    st.header("Subject Progress")

    subjects = topics_df["Subject"].unique() if not topics_df.empty else []
    if len(subjects) == 0:
        st.warning("No subjects found. Add topics first.")

    for sub in subjects:

        subject_df = topics_df[topics_df["Subject"] == sub]

        total = len(subject_df)

        Completed = len(
            subject_df[
                subject_df["Status"].str.strip().str.lower() == "Completed"
                ]
        )

        in_progress = len(
            subject_df[
                subject_df["Status"].str.strip().str.lower() == "in progress"
                ]
        )

        progress = (
                           Completed + (0.5 * in_progress)
                   ) / total if total > 0 else 0

        st.write(sub)

        st.progress(progress)
        st.write(f"{round(progress * 100, 1)}%")

elif page == "Study Tracker":

    st.header("Study Tracker")

    subject = st.selectbox(
        "Select Subject",
        [
            "History",
            "Polity",
            "Economy",
            "Geography",
            "Science",
            "Current Affairs"
        ]
    )

    topic = st.text_input("Topic Name")

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ]
    )

    if st.button("Save Topic"):

        new_data = pd.DataFrame({
            "Subject": [subject],
            "Topic": [topic],
            "Difficulty": [difficulty],
            "Status": ["Not Started"]
        })

        if os.path.exists(file_path):
            existing = pd.read_csv(file_path, sep="|")

            # REMOVE DUPLICATES BASED ON Subject + Topic
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Subject", "Topic"], keep="first")

            combined.to_csv(file_path, index=False, sep="|")
        else:
            new_data.to_csv(file_path, index=False, sep="|")

        st.success(f"Topic '{topic}' saved successfully!")

    if os.path.exists("data/topics.csv"):
        saved_topics = pd.read_csv("data/topics.csv", sep="|")
    else:
        saved_topics = pd.DataFrame(columns=["Subject", "Topic", "Difficulty", "Status"])

    st.subheader("Saved Topics")

    st.dataframe(
        saved_topics,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Update Topic Status")

    if saved_topics.empty:
        st.warning("No topics available to update")
        selected_topic = None
    else:
        selected_topic = st.selectbox(
            "Select Topic to Update",
            saved_topics["Topic"].unique()
        )

    new_status = st.selectbox(
        "Change Status",
        ["Not Started", "in progress", "Completed"]
    )

    if st.button("Update Status") and selected_topic is not None:
        saved_topics.loc[
            saved_topics["Topic"] == selected_topic,
            "Status"
        ] = new_status

        saved_topics.to_csv("data/topics.csv", index=False, sep="|")

        st.success(f"Status updated to '{new_status}' for {selected_topic}")

    st.subheader("Delete Topic")

    confirm_delete = st.checkbox(
        "I confirm deletion"
    )

    if selected_topic is not None:

        if st.button("🗑 Delete Topic") and confirm_delete:
            saved_topics = saved_topics[
                saved_topics["Topic"] != selected_topic
                ]

            saved_topics.to_csv("data/topics.csv", index=False, sep="|")

            st.success(
                f"Topic '{selected_topic}' deleted successfully!"
            )

            st.rerun()


elif page == "Quiz":

    st.header("TNPSC Quiz")

    st.subheader("🤖 AI Quiz Generator")

    quiz_topic = st.text_input(
        "Enter Quiz Topic",
        key="ai_quiz_topic"
    )

    num_questions = st.selectbox(
        "Number of Questions",
        [5, 10, 15, 20],
        key="ai_quiz_count"
    )

    if st.button("Generate AI Quiz"):

        if quiz_topic.strip() == "":
            st.warning("Please enter a topic")
        else:

            with st.spinner("Generating TNPSC Quiz..."):

                prompt = f"""
                You are a TNPSC Group 1 question setter.

                Generate exactly {num_questions} questions on:

                {quiz_topic}

                Return ONLY valid JSON.

                Format:

                [
                  {{
                    "question":"Question text",
                    "optionA":"OptionA",
                    "optionB":"OptionB",
                    "optionC":"OptionC",
                    "optionD":"OptionD",
                    "answer":"A",
                    "explanation":"Explanation text"
                  }}
                ]

                Rules:
                - TNPSC Group 1 level
                - 4 options
                - One correct answer
                - Detailed explanation
                - Return JSON only
                - Use double quotes only
                - Do not use apostrophes inside JSON values
                - Return JSON array only
                """

            import json
            import re

            result = ask_gemini(prompt)

            cleaned = result.replace("```json", "").replace("```", "").strip()

            try:

                match = re.search(r"\[.*\]", cleaned, re.DOTALL)

                if match:

                    json_text = match.group()

                    quiz_data = json.loads(json_text)

                    st.session_state.ai_quiz = quiz_data
                    st.session_state.ai_quiz_index = 0
                    st.session_state.ai_quiz_score = 0

                    st.success("AI Quiz Generated Successfully ✅")

                else:

                    st.error("No valid JSON found")

                    st.text_area(
                        "AI Response",
                        cleaned,
                        height=400
                    )

            except Exception as e:

                st.error(f"JSON Parse Error: {e}")

                st.text_area(
                    "AI Response",
                    cleaned,
                    height=400
                )

    st.divider()
    if "ai_quiz" in st.session_state and st.session_state.ai_quiz:

        quiz = st.session_state.ai_quiz
        index = st.session_state.ai_quiz_index

        if index < len(quiz):

            q = quiz[index]

            st.subheader(f"AI Question {index + 1}/{len(quiz)}")
            st.progress((index + 1) / len(quiz))

            st.write(q["question"])

            options = [
                q["optionA"],
                q["optionB"],
                q["optionC"],
                q["optionD"]
            ]

            selected = st.radio(
                "Choose Answer",
                options,
                key=f"ai_{index}"
            )

            if st.button("Submit AI Answer"):

                options_map = {
                    "A": q["optionA"],
                    "B": q["optionB"],
                    "C": q["optionC"],
                    "D": q["optionD"]
                }

                correct_option = options_map[q["answer"]]

                if selected == correct_option:
                    st.success("Correct ✅")
                    st.session_state.ai_quiz_score += 1
                else:
                    st.error(f"Wrong ❌ Correct Answer: {correct_option}")

                st.info(f"Explanation: {q['explanation']}")

            if st.button("Next AI Question"):
                st.session_state.ai_quiz_index += 1

                st.rerun()

        else:

            st.success(
                f"""
    AI Quiz Completed!

    Score:
    {st.session_state.ai_quiz_score}
    /
    {len(quiz)}
    """
            )

            percentage = (
                                 st.session_state.ai_quiz_score
                                 / len(quiz)
                         ) * 100

            st.write(
                f"Percentage: {round(percentage, 2)}%"
            )

            if percentage >= 80:
                st.success("Excellent TNPSC Preparation 🔥")

            elif percentage >= 60:
                st.info("Good Progress 👍")

            else:
                st.warning("Needs More Practice 📚")

            if st.button("Restart AI Quiz"):
                del st.session_state.ai_quiz
                st.session_state.ai_quiz_index = 0
                st.session_state.ai_quiz_score = 0

                st.rerun()

    st.divider()

    st.subheader("📜 PYQ Quiz Generator")

    st.write(
            "Practice TNPSC Previous Year Questions with explanations and score tracking."
        )

    if questions_df.empty:
        st.warning("No questions found in questions.csv")
    else:

        subject_list = questions_df["Subject"].dropna().unique()

        if len(subject_list) == 0:
            st.warning("No subjects found in questions.csv")
            st.stop()

        subject = st.selectbox("Choose Subject", subject_list)

        subject_questions = questions_df[
            questions_df["Subject"] == subject
        ]

        if "quiz_index" not in st.session_state:
            st.session_state.quiz_index = 0

        if "quiz_score" not in st.session_state:
            st.session_state.quiz_score = 0

        if "quiz_data" not in st.session_state:
            st.session_state.quiz_data = None

        if st.button("Start Quiz"):

            st.session_state.quiz_data = subject_questions.sample(
                min(20, len(subject_questions))
            ).reset_index(drop=True)

            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0

            st.rerun()

        if st.session_state.quiz_data is not None:

            quiz = st.session_state.quiz_data
            index = st.session_state.quiz_index

            if index < len(quiz):

                row = quiz.iloc[index]

                st.subheader(
                    f"Question {index + 1} / {len(quiz)}"
                )

                st.write(row["Question"])

                options = [
                    row["OptionA"],
                    row["OptionB"],
                    row["OptionC"],
                    row["OptionD"]
                ]

                answer = st.radio(
                    "Choose Answer",
                    options,
                    key=f"q{index}"
                )

                if st.button("Submit Answer"):

                    if answer == row["Answer"]:

                        st.success("Correct ✅")

                        st.session_state.quiz_score += 1

                    else:

                        st.error(
                            f"Wrong ❌ Correct Answer: {row['Answer']}"
                        )

                    st.info(
                        f"Explanation: {row['Explanation']}"
                    )



                if st.button("Next Question"):

                    st.session_state.quiz_index += 1

                    st.rerun()

            else:

                st.success(
                    f"""
                    Quiz Completed!

                    Score:
                    {st.session_state.quiz_score}
                    / {len(quiz)}
                    """
                )

                percentage = (
                    st.session_state.quiz_score
                    / len(quiz)
                ) * 100

                st.write(
                    f"Percentage: {round(percentage,2)}%"
                )

                if percentage >= 80:
                    st.success("Excellent TNPSC Preparation 🔥")

                elif percentage >= 60:
                    st.info("Good Progress 👍")

                else:
                    st.warning("Needs More Practice 📚")

                if st.button("Restart Quiz"):

                    st.session_state.quiz_data = None
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_score = 0

                    st.rerun()

elif page == "Report":

    st.header("Performance Report")

    import matplotlib.pyplot as plt

    # 🚨 SAFETY CHECK (MOST IMPORTANT)
    if topics_df.empty:
        st.warning("No topics available. Please add topics in Study Tracker first.")
    else:

        # -----------------------------
        # 1. Topics per Subject
        # -----------------------------
        st.subheader("Topics per Subject")

        subject_counts = topics_df["Subject"].fillna("Unknown").value_counts()

        fig, ax = plt.subplots(figsize=(5, 3))

        subject_counts.plot(kind="bar", ax=ax)

        ax.set_ylabel("Number of Topics")
        ax.set_ylim(0, subject_counts.max() * 1.2)

        if len(subject_counts) <= 15:
            for i, v in enumerate(subject_counts.values):
                ax.text(i, v, str(v), ha="center", fontsize=8)

        st.pyplot(fig, use_container_width=False)

        # -----------------------------
        # 2. Status Distribution
        # -----------------------------
        st.subheader("Study Status Overview")

        status_counts = topics_df["Status"].fillna("Not Started").value_counts()

        fig2, ax2 = plt.subplots(figsize=(5, 3))

        status_counts.plot(kind="bar", ax=ax2)

        ax2.set_ylabel("Count")
        ax2.set_ylim(0, status_counts.max() * 1.2)

        if len(status_counts) <= 15:
            for i, v in enumerate(status_counts.values):
                ax2.text(i, v, str(v), ha="center", fontsize=8)

        st.pyplot(fig2, use_container_width=False)
