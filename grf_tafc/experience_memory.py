"""Key-value experience memory with embedding retrieval (Eq. 10) and de-duplicated update (Eq. 14)."""
import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

Vector = List[float]


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class GeminiEmbedder:
    """Text embeddings from the Gemini embedding API, cached on disk to save quota."""

    def __init__(self, client, model: str, cache_path: Optional[Path] = None):
        from google.genai import types
        self._types = types
        self.client, self.model, self.cache_path = client, model, cache_path
        self.cache = {}
        if cache_path and Path(cache_path).exists():
            self.cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))

    def _k(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        missing = [t for t in dict.fromkeys(texts) if self._k(t) not in self.cache]
        if missing:
            res = self.client.models.embed_content(
                model=self.model, contents=missing,
                config=self._types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
            for t, e in zip(missing, res.embeddings):
                self.cache[self._k(t)] = list(e.values)
            if self.cache_path:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.cache_path).write_text(json.dumps(self.cache), encoding="utf-8")
        return [self.cache[self._k(t)] for t in texts]


class HashingEmbedder:
    """Offline bag-of-words embedder (no API calls). Useful for tests or as a placeholder."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(v)
        return out


def cosine(a: Vector, b: Vector) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@dataclass
class ExperienceEntry:
    key: str      # applicable task context
    value: str    # recommended formation adjustment + supporting rationale
    source: str = "reflection"    # seed / reflection / manual
    created_at: str = ""


# Optional hook: (new_value, existing_value) -> True if they describe an equivalent adjustment.
EquivalenceJudge = Callable[[str, str], bool]


class ExperienceMemory:
    def __init__(self, path: Path, embedder, seed_path: Optional[Path] = None,
                 dedup_key_similarity: float = 0.85, dedup_value_similarity: float = 0.92,
                 equivalence_judge: Optional[EquivalenceJudge] = None):
        self.path, self.embedder = Path(path), embedder
        self.dedup_key_similarity = dedup_key_similarity
        self.dedup_value_similarity = dedup_value_similarity
        self.equivalence_judge = equivalence_judge
        if not self.path.exists() and seed_path and Path(seed_path).exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(seed_path, self.path)
        self.entries: List[ExperienceEntry] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [ExperienceEntry(**{k: e.get(k, "") for k in ("key", "value", "source", "created_at")})
                            for e in data.get("entries", [])]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"entries": [asdict(e) for e in self.entries]},
                                        ensure_ascii=False, indent=2), encoding="utf-8")

    # Eq. (10): rank stored keys by cosine similarity to the current task context
    def retrieve(self, query: str, top_k: int = 2, min_similarity: float = 0.0
                 ) -> List[Tuple[ExperienceEntry, float]]:
        if not self.entries:
            return []
        vecs = self.embedder.embed([query] + [e.key for e in self.entries])
        scored = [(e, cosine(vecs[0], v)) for e, v in zip(self.entries, vecs[1:])]
        scored.sort(key=lambda s: s[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] >= min_similarity]

    # Eq. (14): M <- U(M, insight) with semantic de-duplication
    def update(self, key: str, value: str, source: str = "reflection") -> Tuple[bool, str]:
        similar = self.retrieve(key, top_k=len(self.entries), min_similarity=self.dedup_key_similarity)
        if similar:
            vals = self.embedder.embed([value] + [e.value for e, _ in similar])
            for (entry, ksim), v in zip(similar, vals[1:]):
                same = (self.equivalence_judge(value, entry.value) if self.equivalence_judge
                        else cosine(vals[0], v) >= self.dedup_value_similarity)
                if same:
                    return False, f"skipped: equivalent to existing entry (key similarity {ksim:.2f}): {entry.key}"
        self.entries.append(ExperienceEntry(key, value, source, time.strftime("%Y-%m-%d %H:%M:%S")))
        self.save()
        return True, "inserted"
