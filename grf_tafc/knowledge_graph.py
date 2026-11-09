"""Domain knowledge stored as a graph, with seed-entity lookup and multi-hop retrieval (Sec. 3.2.2)."""
import difflib
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional


class DomainKnowledgeGraph:
    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.entities: Dict[str, dict] = {e["id"]: e for e in data["entities"]}
        self.relations: List[dict] = data["relations"]

    # -- seed entity -------------------------------------------------------
    def find_entity(self, query: str) -> Optional[str]:
        """Match a free-text seed entity against entity names / aliases."""
        q = query.strip().lower()
        names = {}
        for eid, e in self.entities.items():
            for n in [e["name"], eid, *e.get("aliases", [])]:
                names[n.lower()] = eid
        if q in names:
            return names[q]
        for n, eid in names.items():           # substring match, e.g. "the ambulance"
            if n in q or q in n:
                return eid
        close = difflib.get_close_matches(q, list(names), n=1, cutoff=0.6)
        return names[close[0]] if close else None

    # -- multi-hop retrieval -----------------------------------------------
    def explore(self, entity: str, relation: str = "", max_hops: int = 2) -> dict:
        seed = self.find_entity(entity)
        if seed is None:
            return {"seed": None,
                    "message": f"No entity matches '{entity}'.",
                    "available_entities": [e["name"] for e in self.entities.values()]}
        rel_filter = {r.strip().upper() for r in relation.split(",") if r.strip()}
        triples, knowledge, seen = [], [], {seed}
        frontier = deque([(seed, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if self.entities[node].get("knowledge"):
                knowledge.append(self.entities[node]["knowledge"])
            if depth >= max_hops:
                continue
            for r in self.relations:   # follow edges in both directions
                if rel_filter and r["relation"] not in rel_filter:
                    continue
                if node not in (r["head"], r["tail"]):
                    continue
                triple = (self._name(r["head"]), r["relation"], self._name(r["tail"]))
                if triple not in triples:
                    triples.append(triple)
                nxt = r["tail"] if r["head"] == node else r["head"]
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, depth + 1))
        return {
            "seed": self._name(seed),
            "triples": [{"head": h, "relation": rel, "tail": t} for h, rel, t in triples],
            "knowledge": knowledge,
        }

    def _name(self, eid: str) -> str:
        return self.entities[eid]["name"]
