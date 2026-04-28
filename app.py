# Streamlit UI for Quiz Generator.

import streamlit as st
import random
import os
import json

from pdf_parser import parse_pdf
from embeddings import embed_chunks, cluster_chunks
from knowledge_tracker import KnowledgeTracker
from question_generator import generate_question, grade_answer, OLLAMA_MODEL, MLX_MODEL

st.set_page_config(page_title="Adaptive Quiz Generator", layout="centered")
st.title("Adaptive Quiz Generator")

SAVE_DIR = "saved"

def save_document(name, pdf_bytes, chunks):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(f"{SAVE_DIR}/{name}.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open(f"{SAVE_DIR}/{name}.json", "w") as f:
        json.dump(chunks, f)

def get_saved_docs():
    if not os.path.exists(SAVE_DIR):
        return []
    return [f[:-5] for f in os.listdir(SAVE_DIR) if f.endswith(".json")]

def load_saved_doc(name):
    with open(f"{SAVE_DIR}/{name}.json") as f:
        return json.load(f)

# session state setup
if "phase"          not in st.session_state: st.session_state.phase          = "setup"
if "chunks"         not in st.session_state: st.session_state.chunks         = None
if "tracker"        not in st.session_state: st.session_state.tracker        = None
if "question"       not in st.session_state: st.session_state.question       = None
if "topic_id"       not in st.session_state: st.session_state.topic_id       = None
if "answered"       not in st.session_state: st.session_state.answered       = False
if "last_correct"   not in st.session_state: st.session_state.last_correct   = None
if "score"          not in st.session_state: st.session_state.score          = 0
if "total"          not in st.session_state: st.session_state.total          = 0
if "q_queue"        not in st.session_state: st.session_state.q_queue        = []
if "open_feedback"  not in st.session_state: st.session_state.open_feedback  = ""
if "open_reference" not in st.session_state: st.session_state.open_reference = ""
if "open_score"     not in st.session_state: st.session_state.open_score     = 0

with st.sidebar:
    st.header("Settings")

    backend = st.radio("LLM Backend", ["Ollama", "MLX (Apple Silicon)"])
    if backend == "Ollama":
        st.caption(f"Model: `{OLLAMA_MODEL}` — run `ollama serve` first")
    else:
        st.caption(f"Model: `{MLX_MODEL}`")

    mc_count   = st.slider("Multiple choice questions", 1, 15, 7)
    open_count = st.slider("Open-ended questions", 0, 10, 3)
    st.caption(f"Total: {mc_count + open_count} questions")

    if st.session_state.tracker and st.session_state.total > 0:
        st.divider()
        st.subheader("Topic Mastery")
        for row in st.session_state.tracker.get_stats():
            if row["attempts"] > 0:
                filled = int(row["mastery"] / 10)
                bar = "█" * filled + "░" * (10 - filled)
                st.text(f"{row['topic']}: {bar} {row['mastery']}%")

    if st.session_state.phase != "setup" and st.button("Reset"):
        for k in ["phase", "chunks", "tracker", "question", "topic_id",
                  "answered", "last_correct", "score", "total", "q_queue",
                  "open_feedback", "open_reference", "open_score"]:
            del st.session_state[k]
        st.rerun()

backend_key = "ollama" if backend == "Ollama" else "mlx"
quiz_length = mc_count + open_count

def start_quiz(chunks):
    embeddings = embed_chunks(chunks)
    labels     = cluster_chunks(embeddings, n_topics=5)
    chunk_ids  = [c["id"] for c in chunks]

    q_queue = ["mc"] * mc_count + ["open"] * open_count
    random.shuffle(q_queue)

    st.session_state.chunks  = chunks
    st.session_state.tracker = KnowledgeTracker(labels, chunk_ids)
    st.session_state.q_queue = q_queue
    st.session_state.score   = 0
    st.session_state.total   = 0
    st.session_state.phase   = "quiz"

def continue_quiz():
    # keep the existing tracker so mastery scores carry over
    q_queue = ["mc"] * mc_count + ["open"] * open_count
    random.shuffle(q_queue)

    st.session_state.q_queue        = q_queue
    st.session_state.question       = None
    st.session_state.topic_id       = None
    st.session_state.answered       = False
    st.session_state.last_correct   = None
    st.session_state.score          = 0
    st.session_state.total          = 0
    st.session_state.open_feedback  = ""
    st.session_state.open_reference = ""
    st.session_state.open_score     = 0
    st.session_state.phase          = "quiz"

# setup screen
if st.session_state.phase == "setup":
    saved_docs = get_saved_docs()

    if saved_docs:
        st.subheader("Saved documents")
        selected = st.selectbox("Pick a saved document", saved_docs)
        if st.button("Load & Start Quiz", type="primary"):
            with st.spinner("Loading document..."):
                chunks = load_saved_doc(selected)
            with st.spinner("Embedding and clustering topics..."):
                start_quiz(chunks)
            st.rerun()

        st.divider()

    st.subheader("Upload a new document")
    pdf = st.file_uploader("Upload a PDF to study", type=["pdf"])

    if pdf:
        if st.button("Start Quiz", type="primary"):
            pdf_bytes = pdf.read()

            with st.spinner("Parsing PDF..."):
                chunks = parse_pdf(pdf_bytes)
                if not chunks:
                    st.error("Couldn't read text from this PDF.")
                    st.stop()

            doc_name = pdf.name.rsplit(".", 1)[0]
            save_document(doc_name, pdf_bytes, chunks)

            with st.spinner("Embedding and clustering topics..."):
                start_quiz(chunks)
            st.rerun()

# quiz screen
elif st.session_state.phase == "quiz":
    total = st.session_state.total

    if total >= quiz_length:
        st.session_state.phase = "done"
        st.rerun()

    q_type = st.session_state.q_queue[total]
    label  = "Multiple Choice" if q_type == "mc" else "Open Ended"

    st.progress(total / quiz_length, text=f"Question {total + 1} / {quiz_length} — {label}")

    if st.session_state.question is None:
        with st.spinner("Generating question..."):
            q = None
            for _ in range(5):
                tid  = st.session_state.tracker.pick_next_topic()
                cid  = st.session_state.tracker.pick_chunk(tid)
                text = st.session_state.chunks[cid]["text"]
                q    = generate_question(text, q_type=q_type, backend=backend_key)
                if q:
                    break

        if not q:
            st.error("Couldn't generate a question. Is your model running?")
            if st.button("Try again"):
                st.rerun()
            st.stop()

        st.session_state.question = q
        st.session_state.topic_id = tid
        st.session_state.answered = False

    q   = st.session_state.question
    tid = st.session_state.topic_id

    st.markdown(f"**Topic {tid + 1}**")
    st.markdown(f"### {q['question']}")

    if q["type"] == "mc":
        if not st.session_state.answered:
            choice = st.radio(
                "Choose an answer:",
                options=list(q["options"].keys()),
                format_func=lambda k: f"{k}: {q['options'][k]}",
                index=None,
                key=f"q{total}",
            )
            if st.button("Submit", disabled=(choice is None)):
                correct = (choice == q["answer"])
                st.session_state.tracker.record_answer(tid, correct)
                st.session_state.last_correct = correct
                if correct:
                    st.session_state.score += 1
                st.session_state.total   += 1
                st.session_state.answered = True
                st.rerun()
        else:
            letter = q["answer"]
            if st.session_state.last_correct:
                st.success(f"Correct! {letter}: {q['options'][letter]}")
            else:
                st.error(f"Wrong. The answer was {letter}: {q['options'][letter]}")
            if q.get("explanation"):
                st.info(q["explanation"])
            st.caption(f"Topic {tid + 1} mastery: {st.session_state.tracker.mastery[tid] * 100:.0f}%")
            if st.button("Next question →"):
                st.session_state.question = None
                st.rerun()

    else:
        if not st.session_state.answered:
            user_answer = st.text_area("Your answer:", key=f"open{total}")
            if st.button("Submit", disabled=(not user_answer.strip())):
                with st.spinner("Grading..."):
                    result = grade_answer(q["question"], q["answer"], user_answer, backend=backend_key)
                grade    = result["score"]        if result else 0
                feedback = result.get("feedback", "") if result else ""
                correct  = grade >= 60
                st.session_state.tracker.record_answer(tid, correct)
                st.session_state.last_correct   = correct
                st.session_state.open_score     = grade
                st.session_state.open_feedback  = feedback
                st.session_state.open_reference = q["answer"]
                if correct:
                    st.session_state.score += 1
                st.session_state.total   += 1
                st.session_state.answered = True
                st.rerun()
        else:
            grade = st.session_state.get("open_score", 0)
            if grade >= 80:
                st.success(f"Great answer! {grade}/100")
            elif grade >= 60:
                st.warning(f"Partially correct. {grade}/100")
            else:
                st.error(f"Incorrect. {grade}/100")
            if st.session_state.open_feedback:
                st.info(st.session_state.open_feedback)
            st.markdown(f"**Reference answer:** {st.session_state.open_reference}")
            if st.button("Next question →"):
                st.session_state.question = None
                st.rerun()

# results screen
elif st.session_state.phase == "done":
    score = st.session_state.score
    total = st.session_state.total

    st.header("Quiz finished!")
    st.metric("Score", f"{score} / {total} ({score / total * 100:.0f}%)")

    st.subheader("Topic breakdown")
    weak_topics = []
    for row in st.session_state.tracker.get_stats():
        if row["attempts"] > 0:
            icon = "🟢" if row["mastery"] >= 70 else ("🟡" if row["mastery"] >= 40 else "🔴")
            st.write(f"{icon} **{row['topic']}** — {row['mastery']}% mastery ({row['correct']}/{row['attempts']} correct)")
            if row["mastery"] < 50:
                weak_topics.append(row["topic"])

    st.divider()

    if weak_topics:
        st.write(f"You struggled with: **{', '.join(weak_topics)}**. Practice them now?")
        col1, col2 = st.columns(2)
        if col1.button("Practice weak topics", type="primary"):
            continue_quiz()
            st.rerun()
        if col2.button("Start fresh"):
            for k in ["phase", "chunks", "tracker", "question", "topic_id",
                      "answered", "last_correct", "score", "total", "q_queue",
                      "open_feedback", "open_reference", "open_score"]:
                del st.session_state[k]
            st.rerun()
    else:
        st.success("Great job — no weak topics!")
        if st.button("Take another quiz", type="primary"):
            for k in ["phase", "chunks", "tracker", "question", "topic_id",
                      "answered", "last_correct", "score", "total", "q_queue",
                      "open_feedback", "open_reference", "open_score"]:
                del st.session_state[k]
            st.rerun()
