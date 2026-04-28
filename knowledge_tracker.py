# Track per-topic mastery and adaptively pick the next topic.

import numpy as np


class KnowledgeTracker:

    def __init__(self, topic_labels, chunk_ids):
        # topic_labels: cluster label per chunk; chunk_ids: corresponding chunk ids.
        self.n_topics = int(max(topic_labels)) + 1

        # Map each topic → list of chunk ids that belong to it.
        self.topic_chunks = {}
        for i, label in enumerate(topic_labels):
            if label not in self.topic_chunks:
                self.topic_chunks[label] = []
            self.topic_chunks[label].append(chunk_ids[i])

        self.attempts  = {t: 0   for t in range(self.n_topics)}
        self.correct   = {t: 0   for t in range(self.n_topics)}
        self.mastery   = {t: 0.0 for t in range(self.n_topics)}  # 0.0–1.0
        self.last_seen = {t: -5  for t in range(self.n_topics)}  # question index
        self.q_count   = 0

    def record_answer(self, topic_id, is_correct):
        # Update mastery using an exponential moving average (alpha=0.35).
        self.attempts[topic_id] += 1
        if is_correct:
            self.correct[topic_id] += 1

        alpha = 0.35  # higher = more weight on recent answers
        outcome = 1.0 if is_correct else 0.0
        if self.attempts[topic_id] == 1:
            self.mastery[topic_id] = outcome
        else:
            self.mastery[topic_id] = (1 - alpha) * self.mastery[topic_id] + alpha * outcome

        self.last_seen[topic_id] = self.q_count
        self.q_count += 1

    def pick_next_topic(self):
        # Sample a topic weighted by weakness (+unseen bonus, -recency cooldown).
        weights = []
        for t in range(self.n_topics):
            w = 1.0 - self.mastery[t]          # weaker topics score higher
            if self.attempts[t] == 0:
                w += 0.5                        # bonus for topics not yet seen
            questions_ago = self.q_count - self.last_seen[t]
            if questions_ago < 3:
                w -= 0.3 * (1 - questions_ago / 3)  # cooldown if asked recently
            weights.append(max(w, 0.05))        # floor keeps all topics reachable

        weights = np.array(weights)
        weights /= weights.sum()
        return int(np.random.choice(self.n_topics, p=weights))

    def pick_chunk(self, topic_id):
        return int(np.random.choice(self.topic_chunks[topic_id]))

    def get_stats(self):
        stats = []
        for t in range(self.n_topics):
            stats.append({
                "topic": f"Topic {t + 1}",
                "attempts": self.attempts[t],
                "correct": self.correct[t],
                "mastery": round(self.mastery[t] * 100, 1),
            })
        stats.sort(key=lambda x: x["mastery"])
        return stats
