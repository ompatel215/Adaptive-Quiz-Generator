# Generate MC and open-ended questions from text via a local LLM.
# Backends: Ollama (llama3.2:3b via HTTP) or MLX (quantised, Apple Silicon only).

import requests
import json
import re

OLLAMA_MODEL = "llama3.2:3b"
MLX_MODEL    = "mlx-community/Llama-3.2-3B-Instruct-4bit"

MCQ_PROMPT = """You are a quiz generator. Your job is to write one multiple choice question that tests understanding of the passage below.

Rules:
- The question must be answerable using ONLY information in the passage.
- Do NOT ask "According to the text..." or "The passage states..." — ask directly about the concept.
- The three wrong options must be plausible but clearly incorrect based on the passage.
- The explanation must reference why the correct answer is right.

Passage:
{text}

Respond with ONLY valid JSON, no extra text, no markdown:
{{
  "question": "your question here",
  "options": {{"A": "correct answer", "B": "wrong but plausible", "C": "wrong but plausible", "D": "wrong but plausible"}},
  "answer": "A",
  "explanation": "brief explanation citing the passage"
}}"""

OPEN_PROMPT = """You are a quiz generator. Your job is to write one short-answer question that tests understanding of the passage below.

Rules:
- Ask about a key concept, process, or relationship — not a minor detail or specific number.
- The question must be answerable using ONLY information in the passage.
- Do NOT ask "According to the text..." or "What does the passage say..." — ask directly about the concept.
- The answer should be 1-3 sentences that explain the concept, not just name it.

Passage:
{text}

Respond with ONLY valid JSON, no extra text, no markdown:
{{
  "question": "your question here",
  "answer": "concise answer based on the passage"
}}"""

# cache the MLX model so it only loads once
mlx_model = None
mlx_tokenizer = None

def get_mlx_model():
    global mlx_model, mlx_tokenizer
    if mlx_model is None:
        from mlx_lm import load
        mlx_model, mlx_tokenizer = load(MLX_MODEL)
    return mlx_model, mlx_tokenizer

def clean_chunk(text):
    lines = text.splitlines()
    # only drop lines that are purely numbers/symbols (slide page numbers, decorators)
    cleaned = [l for l in lines if l.strip() and not re.match(r"^\s*[\d\W]+\s*$", l)]
    return "\n".join(cleaned).strip() or text

def call_ollama(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.8}},
        timeout=90,
    )
    return res.json()["response"].strip()

def call_mlx(prompt):
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = get_mlx_model()

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    sampler = make_sampler(temp=0.8)
    raw = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False, sampler=sampler)

    if formatted in raw:
        raw = raw[len(formatted):]

    return raw.strip()

def parse_response(raw):
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

GRADE_PROMPT = """You are a quiz grader. Compare the student's answer to the reference answer and give a similarity score.

Question: {question}
Reference answer: {reference}
Student answer: {student}

Score from 0 to 100 based on how much of the key idea the student captured. Be generous — partial understanding deserves partial credit.

Respond with ONLY valid JSON:
{{"score": 0-100, "feedback": "one sentence explaining the score"}}"""

def grade_answer(question, reference, student, backend="ollama"):
    # Ask the LLM to score the student's answer 0-100.
    prompt = GRADE_PROMPT.format(question=question, reference=reference, student=student)
    try:
        raw = call_ollama(prompt) if backend == "ollama" else call_mlx(prompt)
        data = parse_response(raw)
        if data and "score" in data:
            data["score"] = max(0, min(100, int(data["score"])))  # clamp to 0-100
            return data
    except Exception as e:
        print(f"grading error: {e}")
    return None

def generate_question(chunk_text, q_type="mc", backend="ollama"):
    text = clean_chunk(chunk_text[:2500])
    prompt = (MCQ_PROMPT if q_type == "mc" else OPEN_PROMPT).format(text=text)

    try:
        raw = call_ollama(prompt) if backend == "ollama" else call_mlx(prompt)
        data = parse_response(raw)

        if data is None:
            return None

        if q_type == "mc":
            if not {"question", "options", "answer"}.issubset(data):
                return None
            if data["answer"] not in data["options"]:
                return None
        else:
            if not {"question", "answer"}.issubset(data):
                return None

        data["type"] = q_type
        return data

    except Exception as e:
        print(f"{backend} error: {e}")
        return None
