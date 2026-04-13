#!/usr/bin/env python3
"""
Cross-Project Trust Portability — Standalone Agent Module

Extracted from holodeck-studio trust_portability.py. Enables trust earned in
one project to carry weight in another. Zero external dependencies (stdlib only).

Based on research from:
- Jøsang (2001): A Logic for Uncertain Probabilities — Subjective Logic
- Jøsang, Hayward & Pope (2006): Trust Network Analysis with Subjective Logic
- Ding et al. (2009): Computing Reputation in Online Social Networks
- Gartner TRiSM (2024): Trust, Risk, Security Management for Agentic AI
- RepuNet (2025): Dynamic Dual-Level Reputation for LLM-based MAS

Design principles:
1. Trust is portable — earned in one repo, recognized in others
2. Attestations are signed — tamper-proof via HMAC-SHA256
3. Foreign trust decays — older attestations count less
4. Inconsistency is detectable — conflicting repo reports are flagged
5. Trust propagates — transitivity via Subjective Logic discount operator
6. Replay attacks prevented — each attestation has a unique fingerprint
"""

import json
import time
import math
import hashlib
import hmac
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field


# ── Constants ───────────────────────────────────────────────────────────────

FLEET_TRUST_KEY: str = "superinstance-fleet-trust-v1"
DEFAULT_IMPORT_FACTOR: float = 0.3
ATTESTATION_STALENESS_SECONDS: float = 30 * 86400  # 30 days
FOREIGN_DECAY_RATE: float = 0.96
MAX_PATH_DEPTH: int = 3
INCONSISTENCY_THRESHOLD: float = 0.4
MIN_CONSENSUS_SOURCES: int = 2
ATTESTATION_MAX_AGE: float = 90 * 86400  # 90 days

TRUST_DIMENSIONS: List[str] = [
    "code_quality",
    "task_completion",
    "collaboration",
    "reliability",
    "innovation",
]

BASE_TRUST: float = 0.3


# ── TrustAttestation — Signed Trust Proof ──────────────────────────────────

@dataclass
class TrustAttestation:
    """A cryptographically signed trust proof for cross-project portability.

    Immutable once signed — any modification invalidates the HMAC-SHA256 signature.
    """

    # Identity
    agent_name: str = ""
    issuer_repo: str = ""
    issuer_id: str = ""

    # Trust data
    dimensions: Dict[str, float] = field(default_factory=dict)
    composite: float = BASE_TRUST

    # Evidence metadata
    event_count: int = 0
    is_meaningful: bool = False
    cross_repo_events: List[str] = field(default_factory=list)

    # Timing
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = never expires

    # Cryptographic
    signature: str = ""
    fingerprint: str = ""

    def __post_init__(self) -> None:
        """Ensure dimensions dict has all trust dimensions populated."""
        for dim in TRUST_DIMENSIONS:
            if dim not in self.dimensions:
                self.dimensions[dim] = BASE_TRUST

    def compute_fingerprint(self) -> str:
        """Compute a unique SHA-256 fingerprint for replay detection."""
        content = self._content_hash_input()
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _content_hash_input(self) -> str:
        """Build canonical string for signing/fingerprinting (deterministic order)."""
        sorted_dims = json.dumps(
            {k: self.dimensions[k] for k in sorted(self.dimensions)},
            separators=(",", ":"),
        )
        sorted_events = json.dumps(sorted(self.cross_repo_events), separators=(",", ":"))
        return (
            f"{self.agent_name}|{self.issuer_repo}|{self.issuer_id}|"
            f"{sorted_dims}|{self.composite}|{self.event_count}|"
            f"{self.is_meaningful}|{sorted_events}|{self.issued_at}|{self.expires_at}"
        )

    def sign(self, key: str = FLEET_TRUST_KEY) -> None:
        """Sign this attestation using HMAC-SHA256."""
        self.fingerprint = self.compute_fingerprint()
        message = self._content_hash_input()
        self.signature = hmac.new(
            key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, key: str = FLEET_TRUST_KEY) -> bool:
        """Verify the attestation's HMAC-SHA256 signature."""
        if not self.signature:
            return False
        current_fingerprint = self.compute_fingerprint()
        if self.fingerprint and self.fingerprint != current_fingerprint:
            return False
        message = self._content_hash_input()
        expected = hmac.new(
            key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if this attestation has expired."""
        if self.expires_at <= 0:
            return False
        now = current_time or time.time()
        return now > self.expires_at

    def age_seconds(self, current_time: Optional[float] = None) -> float:
        """How old this attestation is in seconds."""
        now = current_time or time.time()
        return now - self.issued_at

    def age_days(self, current_time: Optional[float] = None) -> float:
        """How old this attestation is in days."""
        return self.age_seconds(current_time) / 86400.0

    def decayed_weight(self, current_time: Optional[float] = None) -> float:
        """Compute decay factor: FOREIGN_DECAY_RATE^days."""
        days = self.age_days(current_time)
        return FOREIGN_DECAY_RATE ** days

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return {
            "agent_name": self.agent_name,
            "issuer_repo": self.issuer_repo,
            "issuer_id": self.issuer_id,
            "dimensions": dict(self.dimensions),
            "composite": round(self.composite, 6),
            "event_count": self.event_count,
            "is_meaningful": self.is_meaningful,
            "cross_repo_events": list(self.cross_repo_events),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrustAttestation":
        """Deserialize from a dictionary."""
        return cls(
            agent_name=data.get("agent_name", ""),
            issuer_repo=data.get("issuer_repo", ""),
            issuer_id=data.get("issuer_id", ""),
            dimensions=data.get("dimensions", {}),
            composite=data.get("composite", BASE_TRUST),
            event_count=data.get("event_count", 0),
            is_meaningful=data.get("is_meaningful", False),
            cross_repo_events=data.get("cross_repo_events", []),
            issued_at=data.get("issued_at", time.time()),
            expires_at=data.get("expires_at", 0.0),
            signature=data.get("signature", ""),
            fingerprint=data.get("fingerprint", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "TrustAttestation":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))


# ── InconsistencyReport ─────────────────────────────────────────────────────

@dataclass
class InconsistencyReport:
    """A report of trust inconsistency between repos for an agent."""

    agent_name: str
    repo_scores: Dict[str, float]
    max_difference: float
    flagged: bool
    description: str

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "agent_name": self.agent_name,
            "repo_scores": {k: round(v, 4) for k, v in self.repo_scores.items()},
            "max_difference": round(self.max_difference, 4),
            "flagged": self.flagged,
            "description": self.description,
        }


# ── FleetTrustBridge ───────────────────────────────────────────────────────

class FleetTrustBridge:
    """Bridge between local and foreign trust.

    Maintains a cache of foreign trust attestations and blends them with
    local trust scores. Provides unified fleet-wide trust computation.
    """

    def __init__(
        self,
        local_repo: str = "local",
        import_factor: float = DEFAULT_IMPORT_FACTOR,
        fleet_key: str = FLEET_TRUST_KEY,
        trust_getter: Optional[Callable[[str], float]] = None,
    ) -> None:
        self.local_repo: str = local_repo
        self.import_factor: float = max(0.0, min(1.0, import_factor))
        self.fleet_key: str = fleet_key
        self.trust_getter: Optional[Callable[[str], float]] = trust_getter

        self._foreign_attestations: Dict[str, List[TrustAttestation]] = {}
        self._seen_fingerprints: Dict[str, float] = {}
        self._inconsistencies: Dict[str, InconsistencyReport] = {}
        self._import_count: int = 0
        self._replay_count: int = 0
        self._invalid_count: int = 0

    def _get_local_trust(self, agent_name: str) -> float:
        """Get local trust score for an agent."""
        if self.trust_getter:
            try:
                return max(0.0, min(1.0, self.trust_getter(agent_name)))
            except Exception:
                pass
        return BASE_TRUST

    # ── Import / Export ──────────────────────────────────────────

    def import_attestation(
        self,
        attestation: TrustAttestation,
        current_time: Optional[float] = None,
    ) -> dict:
        """Import and validate a foreign trust attestation.

        Checks: signature, replay, expiration, max age.
        """
        now = current_time or time.time()

        if not attestation.verify(self.fleet_key):
            self._invalid_count += 1
            return {"accepted": False, "reason": "invalid_signature", "agent_name": attestation.agent_name}

        fp = attestation.fingerprint or attestation.compute_fingerprint()
        if fp in self._seen_fingerprints:
            self._replay_count += 1
            return {"accepted": False, "reason": "replay_detected", "agent_name": attestation.agent_name, "fingerprint": fp}

        if attestation.is_expired(now):
            self._invalid_count += 1
            return {"accepted": False, "reason": "expired", "agent_name": attestation.agent_name}

        if attestation.age_seconds(now) > ATTESTATION_MAX_AGE:
            self._invalid_count += 1
            return {"accepted": False, "reason": "too_old", "agent_name": attestation.agent_name, "age_days": round(attestation.age_days(now), 1)}

        self._seen_fingerprints[fp] = now
        agent = attestation.agent_name

        if agent not in self._foreign_attestations:
            self._foreign_attestations[agent] = []

        # Keep only the most recent attestation from each issuer
        existing = [a for a in self._foreign_attestations[agent] if a.issuer_repo != attestation.issuer_repo]
        existing.append(attestation)
        self._foreign_attestations[agent] = existing

        self._import_count += 1
        return {"accepted": True, "reason": "valid", "agent_name": agent, "issuer_repo": attestation.issuer_repo, "composite": attestation.composite}

    def export_attestation(
        self,
        agent_name: str,
        trust_getter: Callable[[str], float],
        composite_getter: Callable[[], float],
        event_count_getter: Optional[Callable[[], int]] = None,
        meaningful_getter: Optional[Callable[[], bool]] = None,
        cross_repo_events: Optional[List[str]] = None,
    ) -> TrustAttestation:
        """Create a signed attestation for a local agent's trust."""
        dimensions: Dict[str, float] = {}
        for dim in TRUST_DIMENSIONS:
            try:
                dimensions[dim] = max(0.0, min(1.0, trust_getter(dim)))
            except Exception:
                dimensions[dim] = BASE_TRUST

        composite = max(0.0, min(1.0, composite_getter()))

        event_count = 0
        if event_count_getter:
            try:
                event_count = event_count_getter()
            except Exception:
                pass

        is_meaningful = False
        if meaningful_getter:
            try:
                is_meaningful = meaningful_getter()
            except Exception:
                pass

        att = TrustAttestation(
            agent_name=agent_name,
            issuer_repo=self.local_repo,
            issuer_id=self.local_repo,
            dimensions=dimensions,
            composite=composite,
            event_count=event_count,
            is_meaningful=is_meaningful,
            cross_repo_events=cross_repo_events or [],
        )
        att.sign(self.fleet_key)
        return att

    # ── Trust Blending ──────────────────────────────────────────

    def foreign_trust(self, agent_name: str, current_time: Optional[float] = None) -> float:
        """Compute weighted consensus of foreign trust for an agent."""
        attestations = self._foreign_attestations.get(agent_name, [])
        if not attestations:
            return BASE_TRUST

        now = current_time or time.time()
        weighted_sum = 0.0
        weight_total = 0.0

        for att in attestations:
            if att.is_expired(now):
                continue
            decay = att.decayed_weight(now)
            meaningful_bonus = 1.5 if att.is_meaningful else 1.0
            event_bonus = min(2.0, 1.0 + math.log1p(att.event_count) / math.log1p(20))
            total_weight = decay * meaningful_bonus * event_bonus
            weighted_sum += att.composite * total_weight
            weight_total += total_weight

        if weight_total <= 0:
            return BASE_TRUST
        return max(0.0, min(1.0, weighted_sum / weight_total))

    def fleet_composite_trust(self, agent_name: str, current_time: Optional[float] = None) -> float:
        """Blend local and foreign trust: (1-α)*local + α*foreign."""
        local = self._get_local_trust(agent_name)
        foreign = self.foreign_trust(agent_name, current_time)

        attestations = self._foreign_attestations.get(agent_name, [])
        now = current_time or time.time()
        active = [a for a in attestations if not a.is_expired(now)]
        if not active:
            return local

        blended = (1.0 - self.import_factor) * local + self.import_factor * foreign
        return max(0.0, min(1.0, blended))

    # ── Inconsistency Detection ─────────────────────────────────

    def detect_inconsistencies(self, current_time: Optional[float] = None) -> List[InconsistencyReport]:
        """Detect trust inconsistency across repos for all agents."""
        now = current_time or time.time()
        reports: List[InconsistencyReport] = []

        for agent_name, attestations in self._foreign_attestations.items():
            active = [a for a in attestations if not a.is_expired(now)]
            if len(active) < MIN_CONSENSUS_SOURCES:
                continue

            repo_scores: Dict[str, float] = {att.issuer_repo: att.composite for att in active}
            local = self._get_local_trust(agent_name)
            repo_scores[self.local_repo] = local

            scores = list(repo_scores.values())
            max_diff = max(scores) - min(scores)
            flagged = max_diff > INCONSISTENCY_THRESHOLD

            report = InconsistencyReport(
                agent_name=agent_name,
                repo_scores=repo_scores,
                max_difference=max_diff,
                flagged=flagged,
                description=(
                    f"Trust for '{agent_name}' varies from {min(scores):.2f} to "
                    f"{max(scores):.2f} across {len(repo_scores)} sources"
                    + (" — INCONSISTENT" if flagged else "")
                ),
            )
            reports.append(report)
            self._inconsistencies[agent_name] = report

        return sorted(reports, key=lambda r: r.max_difference, reverse=True)

    # ── Consensus ───────────────────────────────────────────────

    def trust_consensus(self, agent_name: str, current_time: Optional[float] = None) -> dict:
        """Compute trust consensus across all repos for an agent."""
        local = self._get_local_trust(agent_name)
        foreign = self.foreign_trust(agent_name, current_time)
        fleet = self.fleet_composite_trust(agent_name, current_time)

        attestations = self._foreign_attestations.get(agent_name, [])
        now = current_time or time.time()
        sources: Dict[str, dict] = {
            att.issuer_repo: {
                "composite": round(att.composite, 4),
                "age_days": round(att.age_days(now), 1),
                "decayed_weight": round(att.decayed_weight(now), 4),
                "meaningful": att.is_meaningful,
            }
            for att in attestations
            if not att.is_expired(now)
        }
        sources[self.local_repo] = {"composite": round(local, 4), "age_days": 0.0, "decayed_weight": 1.0, "meaningful": True}

        all_scores = [s["composite"] for s in sources.values()]
        if len(all_scores) >= 2:
            mean = sum(all_scores) / len(all_scores)
            variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
            std_dev = math.sqrt(variance)
            consensus_score = max(0.0, 1.0 - std_dev / 0.5)
        else:
            consensus_score = 1.0

        return {
            "agent_name": agent_name,
            "local_trust": round(local, 4),
            "foreign_trust": round(foreign, 4),
            "fleet_trust": round(fleet, 4),
            "source_count": len(sources),
            "sources": sources,
            "consensus_score": round(consensus_score, 4),
        }

    # ── Maintenance ─────────────────────────────────────────────

    def prune_stale_attestations(self, current_time: Optional[float] = None) -> int:
        """Remove expired attestations. Returns count removed."""
        now = current_time or time.time()
        removed = 0

        for agent_name in list(self._foreign_attestations.keys()):
            before = len(self._foreign_attestations[agent_name])
            self._foreign_attestations[agent_name] = [
                a for a in self._foreign_attestations[agent_name]
                if not a.is_expired(now) and a.age_seconds(now) <= ATTESTATION_MAX_AGE
            ]
            after = len(self._foreign_attestations[agent_name])
            removed += before - after
            if not self._foreign_attestations[agent_name]:
                del self._foreign_attestations[agent_name]

        fp_cutoff = now - ATTESTATION_MAX_AGE - (7 * 86400)
        stale_fps = [fp for fp, ts in self._seen_fingerprints.items() if ts < fp_cutoff]
        for fp in stale_fps:
            del self._seen_fingerprints[fp]

        return removed

    def stats(self) -> dict:
        """Bridge statistics."""
        total_attestations = sum(len(atts) for atts in self._foreign_attestations.values())
        return {
            "local_repo": self.local_repo,
            "import_factor": self.import_factor,
            "agents_with_foreign_trust": len(self._foreign_attestations),
            "total_foreign_attestations": total_attestations,
            "total_imports": self._import_count,
            "replays_detected": self._replay_count,
            "invalid_attestations": self._invalid_count,
        }

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the bridge state."""
        return {
            "local_repo": self.local_repo,
            "import_factor": self.import_factor,
            "foreign_attestations": {
                agent: [att.to_dict() for att in atts]
                for agent, atts in self._foreign_attestations.items()
            },
            "stats": self.stats(),
        }
