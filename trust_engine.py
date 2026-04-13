#!/usr/bin/env python3
"""
Multi-Dimensional Trust Engine — Standalone Agent Module

Extracted from holodeck-studio trust_engine.py. Provides the core trust scoring
system for multi-agent fleets. Zero external dependencies (stdlib only).

Based on research from:
- PNAS 2024: Emergent in-group behavior in multi-agent RL
- RepuNet 2025: Dynamic dual-level reputation for LLM multi-agent systems
- ACM 2015: Trust and reputation models survey
- TRiSM (Gartner 2024): Trust, Risk, Security Management for Agentic AI

Design principles:
1. Trust is earned, not assigned — starts at base (0.3)
2. Multiple independent trust dimensions — different skills, different trust
3. Temporal decay — recent behavior matters more than ancient history
4. Context-aware — trust for code quality != trust for social behavior
5. Composite scoring — overall trust is weighted combination of dimensions
"""

import json
import time
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


# ── Trust dimensions — each tracks a different capability area ──────────────

TRUST_DIMENSIONS: List[str] = [
    "code_quality",
    "task_completion",
    "collaboration",
    "reliability",
    "innovation",
]

# Default weights for composite trust calculation
DEFAULT_WEIGHTS: Dict[str, float] = {
    "code_quality": 0.25,
    "task_completion": 0.30,
    "collaboration": 0.20,
    "reliability": 0.15,
    "innovation": 0.10,
}

# Decay rates per dimension (per-day exponential decay).
# reliability decays slowest; innovation decays fastest.
DECAY_RATES: Dict[str, float] = {
    "code_quality": 0.97,
    "task_completion": 0.98,
    "collaboration": 0.96,
    "reliability": 0.99,
    "innovation": 0.93,
}

# Base trust for agents with no history
BASE_TRUST: float = 0.3

# Minimum events before trust score is considered "meaningful"
MIN_EVENTS_FOR_TRUST: int = 3

# Trust level thresholds
TRUST_LEVELS: Dict[str, float] = {
    "unknown": 0.0,
    "minimal": 0.2,
    "established": 0.5,
    "trusted": 0.7,
    "exemplary": 0.85,
}


def trust_level_name(score: float) -> str:
    """Return the human-readable trust level name for a given score."""
    if score >= TRUST_LEVELS["exemplary"]:
        return "exemplary"
    if score >= TRUST_LEVELS["trusted"]:
        return "trusted"
    if score >= TRUST_LEVELS["established"]:
        return "established"
    if score >= TRUST_LEVELS["minimal"]:
        return "minimal"
    return "unknown"


# ── Trust event presets ─────────────────────────────────────────────────────

TRUST_EVENTS: Dict[str, Dict[str, object]] = {
    "task_completed": {"dimension": "task_completion", "value": 0.8, "weight": 1.0},
    "task_completed_excellent": {"dimension": "task_completion", "value": 1.0, "weight": 1.5},
    "task_failed": {"dimension": "reliability", "value": 0.2, "weight": 1.5},
    "code_review_passed": {"dimension": "code_quality", "value": 0.9, "weight": 1.0},
    "code_review_failed": {"dimension": "code_quality", "value": 0.3, "weight": 1.5},
    "collaboration_good": {"dimension": "collaboration", "value": 0.85, "weight": 1.0},
    "conflict_resolved": {"dimension": "collaboration", "value": 0.9, "weight": 1.2},
    "innovation_shown": {"dimension": "innovation", "value": 0.9, "weight": 1.0},
    "bug_found": {"dimension": "code_quality", "value": 0.85, "weight": 0.8},
    "tests_written": {"dimension": "reliability", "value": 0.8, "weight": 0.7},
    "docs_written": {"dimension": "collaboration", "value": 0.75, "weight": 0.6},
}


# ── WeightedHistory ─────────────────────────────────────────────────────────

class WeightedHistory:
    """Tracks trust events with exponential temporal decay."""

    def __init__(self, decay_rate: float = 0.95) -> None:
        """Initialize with a given per-day decay rate."""
        self.decay_rate: float = decay_rate
        self.events: List[Tuple[float, float, float]] = []  # (timestamp, value, weight)

    def add(self, value: float, weight: float = 1.0, timestamp: Optional[float] = None) -> None:
        """Record a trust event."""
        if timestamp is None:
            timestamp = time.time()
        self.events.append((timestamp, max(0.0, min(1.0, value)), weight))

    def score(self) -> float:
        """Calculate current trust score with temporal decay."""
        if not self.events:
            return BASE_TRUST
        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0
        for ts, value, w in self.events:
            days_ago = (now - ts) / 86400.0
            time_weight = self.decay_rate ** days_ago
            weighted_sum += value * w * time_weight
            weight_total += w * time_weight
        if weight_total <= 0:
            return BASE_TRUST
        return max(0.0, min(1.0, weighted_sum / weight_total))

    def event_count(self) -> int:
        """Return number of recorded events."""
        return len(self.events)

    def recent(self, n: int = 10) -> List[dict]:
        """Get most recent N events as dicts."""
        return [
            {
                "timestamp": ts,
                "value": v,
                "weight": w,
                "days_ago": round((time.time() - ts) / 86400, 2),
            }
            for ts, v, w in sorted(self.events, reverse=True)[:n]
        ]

    def prune(self, max_age_days: int = 90) -> None:
        """Remove events older than max_age_days."""
        cutoff = time.time() - (max_age_days * 86400)
        self.events = [(ts, v, w) for ts, v, w in self.events if ts > cutoff]

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return {
            "decay_rate": self.decay_rate,
            "event_count": len(self.events),
            "score": round(self.score(), 6),
            "events": [{"t": ts, "v": v, "w": w} for ts, v, w in self.events[-50:]],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WeightedHistory":
        """Deserialize from a dictionary."""
        wh = cls(decay_rate=data.get("decay_rate", 0.95))
        for e in data.get("events", []):
            wh.events.append((e["t"], e["v"], e["w"]))
        return wh


# ── TrustScore dataclass ────────────────────────────────────────────────────

@dataclass
class TrustScore:
    """A snapshot of trust at a point in time."""
    composite: float
    level: str
    dimensions: Dict[str, float]
    meaningful: bool
    review_exempt: bool
    total_events: int

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "composite": round(self.composite, 4),
            "level": self.level,
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "meaningful": self.meaningful,
            "review_exempt": self.review_exempt,
            "total_events": self.total_events,
        }


# ── TrustProfile ────────────────────────────────────────────────────────────

@dataclass
class TrustProfile:
    """Complete trust profile for an agent."""

    agent_name: str
    dimensions: Dict[str, WeightedHistory] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Initialize missing dimensions."""
        for dim in TRUST_DIMENSIONS:
            if dim not in self.dimensions:
                self.dimensions[dim] = WeightedHistory(decay_rate=DECAY_RATES[dim])

    def record(self, dimension: str, value: float, weight: float = 1.0) -> None:
        """Record a trust event in a specific dimension."""
        if dimension not in self.dimensions:
            self.dimensions[dimension] = WeightedHistory(decay_rate=0.95)
        self.dimensions[dimension].add(value, weight)
        self.last_seen = time.time()

    def score(self, dimension: Optional[str] = None) -> float:
        """Get trust score for a dimension or composite."""
        if dimension:
            return self.dimensions[dimension].score()
        return self.composite()

    def composite(self, weights: Optional[dict] = None) -> float:
        """Calculate weighted composite trust score."""
        w = weights or self.weights
        scores = {d: h.score() for d, h in self.dimensions.items()}
        total_w = sum(w.get(d, 0) for d in scores)
        if total_w <= 0:
            return BASE_TRUST
        return max(0.0, min(1.0, sum(scores[d] * w.get(d, 0) for d in scores) / total_w))

    def is_meaningful(self) -> bool:
        """Has enough events for trust to be meaningful?"""
        total = sum(h.event_count() for h in self.dimensions.values())
        return total >= MIN_EVENTS_FOR_TRUST

    def review_exempt(self) -> bool:
        """Should this agent be exempt from output review?"""
        return self.is_meaningful() and self.composite() > 0.7

    def get_trust_score(self) -> TrustScore:
        """Return a TrustScore snapshot."""
        comp = self.composite()
        return TrustScore(
            composite=comp,
            level=trust_level_name(comp),
            dimensions={d: h.score() for d, h in self.dimensions.items()},
            meaningful=self.is_meaningful(),
            review_exempt=self.review_exempt(),
            total_events=sum(h.event_count() for h in self.dimensions.values()),
        )

    def summary(self) -> dict:
        """Generate a trust summary dict."""
        ts = self.get_trust_score()
        return {
            "agent": self.agent_name,
            **ts.to_dict(),
            "last_seen": self.last_seen,
        }

    def to_dict(self) -> dict:
        """Serialize the full profile."""
        return {
            "agent_name": self.agent_name,
            "dimensions": {d: h.to_dict() for d, h in self.dimensions.items()},
            "weights": self.weights,
            "created": self.created,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrustProfile":
        """Deserialize from a dictionary."""
        profile = cls(agent_name=data["agent_name"])
        profile.weights = data.get("weights", dict(DEFAULT_WEIGHTS))
        profile.created = data.get("created", time.time())
        profile.last_seen = data.get("last_seen", time.time())
        for d, hd in data.get("dimensions", {}).items():
            profile.dimensions[d] = WeightedHistory.from_dict(hd)
        return profile


# ── TrustEngine ─────────────────────────────────────────────────────────────

class TrustEngine:
    """Fleet-wide trust management engine."""

    def __init__(self, data_dir: str = "trust_data") -> None:
        """Initialize with a persistence directory."""
        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, TrustProfile] = {}

    def get_profile(self, agent_name: str) -> TrustProfile:
        """Get or create trust profile for an agent."""
        if agent_name not in self.profiles:
            self.profiles[agent_name] = TrustProfile(agent_name=agent_name)
        return self.profiles[agent_name]

    def record_event(
        self, agent_name: str, dimension: str, value: float, weight: float = 1.0
    ) -> None:
        """Record a trust event for an agent."""
        self.get_profile(agent_name).record(dimension, value, weight)

    def record_preset(self, agent_name: str, event_name: str) -> bool:
        """Record a trust event from a named preset. Returns False if preset not found."""
        preset = TRUST_EVENTS.get(event_name)
        if not preset:
            return False
        self.record_event(
            agent_name,
            dimension=preset["dimension"],  # type: ignore[arg-type]
            value=preset["value"],  # type: ignore[arg-type]
            weight=preset["weight"],  # type: ignore[arg-type]
        )
        return True

    def get_trust(self, agent_name: str, dimension: Optional[str] = None) -> float:
        """Get trust score for an agent (optionally for a specific dimension)."""
        return self.get_profile(agent_name).score(dimension)

    def composite_trust(self, agent_name: str) -> float:
        """Get composite trust score for an agent."""
        return self.get_profile(agent_name).composite()

    def get_trust_score(self, agent_name: str) -> TrustScore:
        """Return a TrustScore snapshot for an agent."""
        return self.get_profile(agent_name).get_trust_score()

    def compare(self, agent_a: str, agent_b: str) -> dict:
        """Compare trust profiles of two agents."""
        prof_a = self.get_profile(agent_a)
        prof_b = self.get_profile(agent_b)
        return {
            "agent_a": prof_a.summary(),
            "agent_b": prof_b.summary(),
            "similarity": self._similarity(prof_a, prof_b),
        }

    def _similarity(self, a: TrustProfile, b: TrustProfile) -> float:
        """Calculate profile similarity (0-1)."""
        scores_a = {d: h.score() for d, h in a.dimensions.items()}
        scores_b = {d: h.score() for d, h in b.dimensions.items()}
        all_dims = set(scores_a) | set(scores_b)
        if not all_dims:
            return 1.0
        sum_sq_diff = sum(
            (scores_a.get(d, BASE_TRUST) - scores_b.get(d, BASE_TRUST)) ** 2
            for d in all_dims
        )
        max_sq_diff = len(all_dims)
        return 1.0 - math.sqrt(sum_sq_diff / max_sq_diff)

    def leaderboard(self, n: int = 10) -> List[dict]:
        """Get top-N agents by composite trust (meaningful profiles only)."""
        profiles = [
            (name, prof.composite())
            for name, prof in self.profiles.items()
            if prof.is_meaningful()
        ]
        profiles.sort(key=lambda x: x[1], reverse=True)
        return [{"agent": name, "trust": round(score, 4)} for name, score in profiles[:n]]

    def save(self, agent_name: str) -> None:
        """Save a profile to disk."""
        profile = self.profiles.get(agent_name)
        if not profile:
            return
        path = self.data_dir / f"{agent_name}.json"
        path.write_text(json.dumps(profile.to_dict(), indent=2))

    def load(self, agent_name: str) -> Optional[TrustProfile]:
        """Load a profile from disk."""
        path = self.data_dir / f"{agent_name}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            profile = TrustProfile.from_dict(data)
            self.profiles[agent_name] = profile
            return profile
        except (json.JSONDecodeError, KeyError):
            return None

    def save_all(self) -> None:
        """Save all profiles to disk."""
        for name in self.profiles:
            self.save(name)

    def load_all(self) -> None:
        """Load all profiles from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path in self.data_dir.glob("*.json"):
            agent_name = path.stem
            self.load(agent_name)

    def prune_stale(self, max_age_days: int = 60) -> int:
        """Prune profiles not seen in N days. Returns count pruned."""
        cutoff = time.time() - (max_age_days * 86400)
        stale = [
            name for name, prof in self.profiles.items() if prof.last_seen < cutoff
        ]
        for name in stale:
            del self.profiles[name]
            path = self.data_dir / f"{name}.json"
            if path.exists():
                path.unlink()
        return len(stale)

    def stats(self) -> dict:
        """Engine statistics."""
        meaningful = [p for p in self.profiles.values() if p.is_meaningful()]
        return {
            "total_profiles": len(self.profiles),
            "meaningful_profiles": len(meaningful),
            "average_trust": (
                round(sum(p.composite() for p in meaningful) / len(meaningful), 4)
                if meaningful
                else BASE_TRUST
            ),
            "review_exempt": sum(1 for p in meaningful if p.review_exempt()),
            "dimensions": len(TRUST_DIMENSIONS),
        }
