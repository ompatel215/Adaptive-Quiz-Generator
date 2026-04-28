# Embed text chunks and cluster them into topics via K-means.

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Loaded once at import time and reused across all quiz sessions.
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    # Return a 2-D numpy array of embeddings, one row per chunk.
    texts = [c["text"] for c in chunks]
    return model.encode(texts, show_progress_bar=False)


def cluster_chunks(embeddings, n_topics=5):
    # Assign each chunk a topic label (0…k-1) via K-means.
    # k is capped at the number of chunks to avoid empty clusters.
    k = min(n_topics, len(embeddings))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(embeddings)
