# Adaptive Quiz Generator

Upload a PDF and get a quiz that adapts to your weak spots. Topics are discovered automatically via K-means clustering on sentence embeddings. Per-topic mastery is tracked with an exponential moving average — weak topics appear more often until you improve.

Supports two local LLM backends: Ollama (cross-platform) or MLX (Apple Silicon).

---

## Install

```bash
pip install -r requirements.txt
```

For MLX (Apple Silicon only):
```bash
pip install mlx-lm
```

---

## Setup

**Ollama (Mac / Linux / Windows)**
```bash
# Install from https://ollama.com, then:
ollama pull llama3.2:3b
ollama serve   # keep running in a separate terminal
```

**MLX (Apple Silicon only)**
```bash
hf download mlx-community/Llama-3.2-3B-Instruct-4bit
```

---

## Run

```bash
streamlit run app.py
```

---

## Project structure

```
app.py                 # Streamlit UI
pdf_parser.py          # PDF text extraction and chunking
embeddings.py          # Sentence embeddings and K-means clustering
knowledge_tracker.py   # EMA mastery tracking and adaptive topic selection
question_generator.py  # MCQ and open-ended question generation via local LLM
requirements.txt
```
