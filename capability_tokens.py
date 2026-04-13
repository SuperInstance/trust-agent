#!/usr/bin/env python3
"""
Capability Token System — Object-Capability Security — Standalone Agent Module

Extracted from holodeck-studio capability_tokens.py. Zero external dependencies.

Based on research from:
- Dennis & Van Horn (1966): Programming Semantics for Multiprogrammed Computations
- Miller (2006): Robust Capability Security Patterns
- Jøsang (2002): Beta Reputation System with Subjective Logic
- RepuNet (2025): Dynamic Dual-Level Reputation for LLM-based MAS
- Gartner TRiSM (2024): Trust, Risk, Security Management for Agentic AI

Key insight:
    Authority is NOT assigned — it is EARNED through demonstrated trust,
    then CONFERRED as a capability token. The token IS the permission.
"""

import time
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


# ── Constants ───────────────────────────────────────────────────────────────

ENDORSEMENT_TRUST_THRESHOLD: float = 0.4
EXERCISE_TRUST_THRESHOLD: float = 0.25
DELEGATION_TRUST_THRESHOLD: float = 0.5
DEFAULT_CAP_DURATION: int = 0  # permanent
TRUST_DECAY_ALERT_THRESHOLD: float = 0.15
BASE_TRUST: float = 0.3


class CapabilityAction(Enum):
    """Actions a capability token can authorize."""
    BUILD_ROOM = "build_room"
    CREATE_ITEM = "create_item"
    SUMMON_NPC = "summon_npc"
    EDIT_ROOM = "edit_room"
    CREATE_ADVENTURE = "create_adventure"
    REVIEW_AGENT = "review_agent"
    MANAGE_VESSEL = "manage_vessel"
    BROADCAST_FLEET = "broadcast_fleet"
    CREATE_SPELL = "create_spell"
    CREATE_TOOL_ROOM = "create_tool_room"
    MANAGE_PERMISSIONS = "manage_permissions"
    EDIT_ANY_ROOM = "edit_any_room"
    CREATE_ITEM_TYPE = "create_item_type"
    CREATE_ROOM_TYPE = "create_room_type"
    DELEGATE = "delegate"
    GOVERN = "govern"
    SHELL = "shell"


# Map permission levels to capabilities they should receive on endowment
LEVEL_CAPABILITIES: Dict[int, List[CapabilityAction]] = {
    0: [],
    1: [],
    2: [
        CapabilityAction.BUILD_ROOM,
        CapabilityAction.CREATE_ITEM,
        CapabilityAction.SUMMON_NPC,
    ],
    3: [
        CapabilityAction.BUILD_ROOM,
        CapabilityAction.CREATE_ITEM,
        CapabilityAction.SUMMON_NPC,
        CapabilityAction.EDIT_ROOM,
        CapabilityAction.CREATE_ADVENTURE,
        CapabilityAction.REVIEW_AGENT,
        CapabilityAction.MANAGE_VESSEL,
        CapabilityAction.DELEGATE,
    ],
    4: [
        CapabilityAction.BUILD_ROOM,
        CapabilityAction.CREATE_ITEM,
        CapabilityAction.SUMMON_NPC,
        CapabilityAction.EDIT_ROOM,
        CapabilityAction.CREATE_ADVENTURE,
        CapabilityAction.REVIEW_AGENT,
        CapabilityAction.MANAGE_VESSEL,
        CapabilityAction.BROADCAST_FLEET,
        CapabilityAction.CREATE_SPELL,
        CapabilityAction.CREATE_TOOL_ROOM,
        CapabilityAction.MANAGE_PERMISSIONS,
        CapabilityAction.EDIT_ANY_ROOM,
        CapabilityAction.CREATE_ITEM_TYPE,
        CapabilityAction.CREATE_ROOM_TYPE,
        CapabilityAction.DELEGATE,
        CapabilityAction.GOVERN,
    ],
    5: list(CapabilityAction),
}


# ── BetaReputation ──────────────────────────────────────────────────────────

@dataclass
class BetaReputation:
    """Beta distribution-based reputation using Jøsang's Subjective Logic.

    Tracks Beta(α, β) distribution:
        α = positive interactions + 1 (prior)
        β = negative interactions + 1 (prior)

    Expected value R = α / (α + β)
    Uncertainty u = 2 / (α + β + 2)
    """

    alpha: float = 1.0
    beta: float = 1.0
    forget_factor: float = 0.995  # per-event decay

    @property
    def expected_value(self) -> float:
        """Point estimate: R = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """How uncertain we are: u = 2 / (α + β + 2)."""
        return 2.0 / (self.alpha + self.beta + 2)

    @property
    def belief(self) -> float:
        """Belief: b = α / (α + β + 2)."""
        return self.alpha / (self.alpha + self.beta + 2)

    @property
    def disbelief(self) -> float:
        """Disbelief: d = β / (α + β + 2)."""
        return self.beta / (self.alpha + self.beta + 2)

    @property
    def opinion(self) -> Tuple[float, float, float]:
        """Subjective Logic triplet (belief, disbelief, uncertainty)."""
        return (self.belief, self.disbelief, self.uncertainty)

    @property
    def evidence_count(self) -> float:
        """Total evidence: α + β - 2 (subtract priors)."""
        return max(0.0, self.alpha + self.beta - 2)

    def update(self, positive: bool, magnitude: float = 1.0) -> None:
        """Update reputation with new evidence."""
        self.alpha = self.alpha * self.forget_factor
        self.beta = self.beta * self.forget_factor
        if positive:
            self.alpha += 1.0 * magnitude
        else:
            self.beta += 1.0 * magnitude
        self.alpha = max(1.0, self.alpha)
        self.beta = max(1.0, self.beta)

    def update_from_score(self, score: float) -> None:
        """Update from a 0-1 score. >0.5 positive, <0.5 negative."""
        if score > 0.5:
            magnitude = (score - 0.5) * 2.0
            self.update(positive=True, magnitude=max(0.1, magnitude))
        elif score < 0.5:
            magnitude = (0.5 - score) * 2.0
            self.update(positive=False, magnitude=max(0.1, magnitude))

    def discount(self, source_reputation: "BetaReputation") -> "BetaReputation":
        """Trust transitivity via Subjective Logic discount operator."""
        b1, d1, _u1 = source_reputation.opinion
        b2, d2, u2 = self.opinion

        b_result = b1 * b2
        d_result = b1 * d2
        u_result = 1.0 - b_result - d_result

        if 0 < u_result < 1:
            alpha = (b_result / u_result) * 2
            beta = (d_result / u_result) * 2
        else:
            alpha, beta = 1.0, 1.0

        return BetaReputation(alpha=max(1.0, alpha), beta=max(1.0, beta), forget_factor=self.forget_factor)

    def fuse(self, other: "BetaReputation") -> "BetaReputation":
        """Consensus (opinion fusion): combine two opinions about same target."""
        return BetaReputation(
            alpha=max(1.0, self.alpha + other.alpha - 2),
            beta=max(1.0, self.beta + other.beta - 2),
            forget_factor=self.forget_factor,
        )

    def is_suspicious(self, uncertainty_threshold: float = 0.5) -> bool:
        """Is trustworthiness too uncertain to act on?"""
        return self.uncertainty >= uncertainty_threshold

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "forget_factor": self.forget_factor,
            "expected_value": round(self.expected_value, 4),
            "uncertainty": round(self.uncertainty, 4),
            "evidence_count": self.evidence_count,
            "opinion": {
                "belief": round(self.belief, 4),
                "disbelief": round(self.disbelief, 4),
                "uncertainty": round(self.uncertainty, 4),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BetaReputation":
        """Deserialize from dictionary."""
        return cls(
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            forget_factor=data.get("forget_factor", 0.995),
        )


# ── CapabilityToken ─────────────────────────────────────────────────────────

@dataclass
class CapabilityToken:
    """An unforgeable capability token that confers a specific permission.

    In OCap: possession of this token IS the authority.
    Tokens can be attenuated (restricted) but not amplified (upgraded).
    Tokens can be delegated but not stolen.
    """

    # Identity
    token_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action: CapabilityAction = CapabilityAction.BUILD_ROOM

    # Ownership
    holder: str = ""
    issuer: str = ""
    source_token_id: str = ""

    # Trust linkage
    trust_at_issue: float = BASE_TRUST

    # Attenuation (restrictions)
    max_uses: int = 0
    use_count: int = 0
    scope: str = ""
    delegate_depth: int = 0
    max_delegate_depth: int = 3
    actions_allowed: List[CapabilityAction] = field(default_factory=list)

    # Lifecycle
    created: float = field(default_factory=time.time)
    expires: float = 0
    revoked: bool = False
    revoked_at: float = 0
    revoked_reason: str = ""

    # Audit trail
    audit_log: List[dict] = field(default_factory=list)

    def is_valid(self, current_time: Optional[float] = None) -> bool:
        """Check if this token is currently valid."""
        if self.revoked:
            return False
        if self.max_uses > 0 and self.use_count >= self.max_uses:
            return False
        now = current_time or time.time()
        if self.expires > 0 and now > self.expires:
            return False
        if self.delegate_depth > self.max_delegate_depth:
            return False
        return True

    def can_exercise(self, action: Optional[CapabilityAction] = None) -> bool:
        """Check if a specific action can be exercised with this token."""
        if not self.is_valid():
            return False
        target_action = action or self.action
        if self.actions_allowed:
            return target_action in self.actions_allowed
        return target_action == self.action

    def exercise(self, action: Optional[CapabilityAction] = None) -> dict:
        """Exercise this capability token."""
        target_action = action or self.action
        if not self.can_exercise(target_action):
            return {
                "success": False,
                "token_id": self.token_id,
                "error": "Token invalid, expired, or action not permitted",
            }

        self.use_count += 1
        self.audit_log.append({
            "event": "exercise",
            "action": target_action.value,
            "by": self.holder,
            "at": time.time(),
            "use_count": self.use_count,
        })

        remaining = (self.max_uses - self.use_count) if self.max_uses > 0 else -1
        return {"success": True, "token_id": self.token_id, "action": target_action.value, "remaining_uses": remaining, "scope": self.scope}

    def attenuate(
        self,
        max_uses: int = 0,
        scope: str = "",
        allowed_actions: Optional[List[CapabilityAction]] = None,
    ) -> "CapabilityToken":
        """Create an attenuated (restricted) copy of this token."""
        new_token = CapabilityToken(
            action=self.action,
            holder="",
            issuer=self.holder,
            source_token_id=self.token_id,
            trust_at_issue=self.trust_at_issue,
            max_uses=max_uses if max_uses > 0 else self.max_uses,
            scope=scope or self.scope,
            delegate_depth=self.delegate_depth + 1,
            max_delegate_depth=self.max_delegate_depth,
            expires=self.expires,
            actions_allowed=allowed_actions or [self.action],
        )

        # Ensure attenuation: new token cannot have MORE uses
        if self.max_uses > 0 and (max_uses == 0 or max_uses > self.max_uses):
            new_token.max_uses = self.max_uses

        # Ensure attenuation: new token cannot have MORE permissions
        if self.actions_allowed and allowed_actions:
            new_token.actions_allowed = [a for a in allowed_actions if a in self.actions_allowed]
        elif self.actions_allowed:
            new_token.actions_allowed = list(self.actions_allowed)

        self.audit_log.append({"event": "attenuate", "into_token": new_token.token_id, "by": self.holder, "at": time.time()})
        return new_token

    def revoke(self, reason: str = "Revoked by issuer") -> None:
        """Revoke this token and all downstream tokens."""
        self.revoked = True
        self.revoked_at = time.time()
        self.revoked_reason = reason
        self.audit_log.append({"event": "revoke", "reason": reason, "at": time.time()})

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "token_id": self.token_id,
            "action": self.action.value,
            "holder": self.holder,
            "issuer": self.issuer,
            "source_token_id": self.source_token_id,
            "trust_at_issue": round(self.trust_at_issue, 4),
            "max_uses": self.max_uses,
            "use_count": self.use_count,
            "scope": self.scope,
            "delegate_depth": self.delegate_depth,
            "max_delegate_depth": self.max_delegate_depth,
            "actions_allowed": [a.value for a in self.actions_allowed],
            "created": self.created,
            "expires": self.expires,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
            "revoked_reason": self.revoked_reason,
            "is_valid": self.is_valid(),
            "audit_entries": len(self.audit_log),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CapabilityToken":
        """Deserialize from dictionary."""
        token = cls(
            token_id=data.get("token_id", uuid.uuid4().hex[:12]),
            action=CapabilityAction(data.get("action", "build_room")),
            holder=data.get("holder", ""),
            issuer=data.get("issuer", ""),
            source_token_id=data.get("source_token_id", ""),
            trust_at_issue=data.get("trust_at_issue", BASE_TRUST),
            max_uses=data.get("max_uses", 0),
            use_count=data.get("use_count", 0),
            scope=data.get("scope", ""),
            delegate_depth=data.get("delegate_depth", 0),
            max_delegate_depth=data.get("max_delegate_depth", 3),
            expires=data.get("expires", 0),
            revoked=data.get("revoked", False),
            revoked_at=data.get("revoked_at", 0),
            revoked_reason=data.get("revoked_reason", ""),
        )
        token.actions_allowed = [CapabilityAction(a) for a in data.get("actions_allowed", [])]
        return token


# ── CapabilityRegistry ──────────────────────────────────────────────────────

class CapabilityRegistry:
    """Central registry of all capability tokens in the system.

    Implements the Gatekeeper pattern from OCap security.
    """

    def __init__(self, data_dir: str = "capability_data") -> None:
        """Initialize with a persistence directory."""
        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tokens: Dict[str, CapabilityToken] = {}  # token_id -> token
        self.agent_tokens: Dict[str, Set[str]] = {}  # agent_name -> set of token_ids
        self.agent_reputation: Dict[str, BetaReputation] = {}
        self.trust_getter: Optional[Callable[[str], float]] = None

    def set_trust_getter(self, getter: Callable[[str], float]) -> None:
        """Set a callback to get agent trust scores."""
        self.trust_getter = getter

    def _get_trust(self, agent_name: str) -> float:
        """Get trust score for an agent."""
        if self.trust_getter:
            try:
                return self.trust_getter(agent_name)
            except Exception:
                pass
        return BASE_TRUST

    def get_reputation(self, agent_name: str) -> BetaReputation:
        """Get or create BetaReputation for an agent."""
        if agent_name not in self.agent_reputation:
            self.agent_reputation[agent_name] = BetaReputation()
        return self.agent_reputation[agent_name]

    def update_reputation(self, agent_name: str, score: float) -> None:
        """Update an agent's BetaReputation from a 0-1 trust score."""
        self.get_reputation(agent_name).update_from_score(score)

    # ── Token Lifecycle ──────────────────────────────────────────

    def issue(
        self,
        action: CapabilityAction,
        holder: str,
        issuer: str = "system",
        trust_at_issue: Optional[float] = None,
        scope: str = "",
        max_uses: int = 0,
        expires: float = 0,
        delegatable: bool = True,
    ) -> CapabilityToken:
        """Issue a new capability token."""
        trust = trust_at_issue if trust_at_issue is not None else self._get_trust(holder)

        token = CapabilityToken(
            action=action,
            holder=holder,
            issuer=issuer,
            trust_at_issue=trust,
            scope=scope,
            max_uses=max_uses,
            expires=expires,
        )

        self.tokens[token.token_id] = token
        if holder not in self.agent_tokens:
            self.agent_tokens[holder] = set()
        self.agent_tokens[holder].add(token.token_id)

        return token

    def revoke(self, token_id: str, reason: str = "Revoked") -> None:
        """Revoke a capability token and all downstream tokens."""
        if token_id not in self.tokens:
            return
        self.tokens[token_id].revoke(reason)
        downstream = self._find_downstream(token_id)
        for ds_token_id in downstream:
            self.tokens[ds_token_id].revoke(f"Upstream token {token_id} revoked: {reason}")

    def _find_downstream(self, token_id: str) -> List[str]:
        """Find all tokens transitively downstream of the given token."""
        direct_children = [tid for tid, t in self.tokens.items() if t.source_token_id == token_id]
        all_downstream = list(direct_children)
        for child_id in direct_children:
            all_downstream.extend(self._find_downstream(child_id))
        return all_downstream

    def delegate(
        self,
        token_id: str,
        new_holder: str,
        from_agent: str,
        max_uses: int = 0,
        scope: str = "",
    ) -> Optional[CapabilityToken]:
        """Delegate a capability token to another agent."""
        if token_id not in self.tokens:
            return None

        source = self.tokens[token_id]
        if not source.is_valid():
            return None

        delegator_trust = self._get_trust(from_agent)
        if delegator_trust < DELEGATION_TRUST_THRESHOLD:
            return None

        holder_trust = self._get_trust(new_holder)
        if holder_trust < ENDORSEMENT_TRUST_THRESHOLD:
            return None

        new_token = source.attenuate(max_uses=max_uses, scope=scope)
        new_token.holder = new_holder
        new_token.issuer = from_agent

        source.audit_log.append({
            "event": "delegate",
            "to": new_holder,
            "into_token": new_token.token_id,
            "by": from_agent,
            "at": time.time(),
        })

        self.tokens[new_token.token_id] = new_token
        if new_holder not in self.agent_tokens:
            self.agent_tokens[new_holder] = set()
        self.agent_tokens[new_holder].add(new_token.token_id)

        return new_token

    # ── Permission Checks ────────────────────────────────────────

    def can_agent(self, agent_name: str, action: CapabilityAction) -> bool:
        """Check if an agent can perform an action via capability tokens."""
        agent_trust = self._get_trust(agent_name)
        if agent_trust < EXERCISE_TRUST_THRESHOLD:
            return False

        token_ids = self.agent_tokens.get(agent_name, set())
        for tid in token_ids:
            token = self.tokens.get(tid)
            if token and token.can_exercise(action):
                return True
        return False

    def exercise(self, agent_name: str, action: CapabilityAction) -> dict:
        """Exercise a capability token for an agent."""
        token_ids = self.agent_tokens.get(agent_name, set())
        for tid in token_ids:
            token = self.tokens.get(tid)
            if token and token.can_exercise(action):
                return token.exercise(action)
        return {"success": False, "error": f"No valid capability token for action '{action.value}'", "agent": agent_name}

    # ── Endowment ────────────────────────────────────────────────

    def endow_on_level_up(
        self,
        agent_name: str,
        old_level: int,
        new_level: int,
        trust_score: Optional[float] = None,
    ) -> List[CapabilityToken]:
        """Endow an agent with capability tokens when they level up."""
        trust = trust_score if trust_score is not None else self._get_trust(agent_name)
        new_tokens: List[CapabilityToken] = []

        for lvl in range(old_level + 1, new_level + 1):
            for action in LEVEL_CAPABILITIES.get(lvl, []):
                if not self.can_agent(agent_name, action):
                    token = self.issue(action=action, holder=agent_name, issuer="level_up", trust_at_issue=trust)
                    new_tokens.append(token)

        return new_tokens

    # ── Agent Summary ────────────────────────────────────────────

    def agent_capabilities(self, agent_name: str) -> List[dict]:
        """Get all capability tokens held by an agent."""
        token_ids = self.agent_tokens.get(agent_name, set())
        return [self.tokens[tid].to_dict() for tid in token_ids if tid in self.tokens]

    def agent_summary(self, agent_name: str) -> dict:
        """Get complete capability summary for an agent."""
        caps = self.agent_capabilities(agent_name)
        valid = [c for c in caps if c["is_valid"]]
        rep = self.get_reputation(agent_name)

        return {
            "agent": agent_name,
            "trust": round(self._get_trust(agent_name), 4),
            "reputation": rep.to_dict(),
            "total_tokens": len(caps),
            "valid_tokens": len(valid),
            "revoked_tokens": len(caps) - len(valid),
            "actions": list(set(c["action"] for c in valid)),
            "capabilities": valid,
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self, agent_name: str) -> None:
        """Save an agent's capability state to disk."""
        caps = self.agent_capabilities(agent_name)
        rep = self.agent_reputation.get(agent_name)
        data = {"agent": agent_name, "capabilities": caps, "reputation": rep.to_dict() if rep else None}
        path = self.data_dir / f"{agent_name}.json"
        path.write_text(json.dumps(data, indent=2))

    def load(self, agent_name: str) -> bool:
        """Load an agent's capability state from disk."""
        path = self.data_dir / f"{agent_name}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            for cap_data in data.get("capabilities", []):
                token = CapabilityToken.from_dict(cap_data)
                self.tokens[token.token_id] = token
                if agent_name not in self.agent_tokens:
                    self.agent_tokens[agent_name] = set()
                self.agent_tokens[agent_name].add(token.token_id)
            rep_data = data.get("reputation")
            if rep_data:
                self.agent_reputation[agent_name] = BetaReputation.from_dict(rep_data)
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def save_all(self) -> None:
        """Save all agents' capability states."""
        for agent_name in self.agent_tokens:
            self.save(agent_name)

    def load_all(self) -> None:
        """Load all agents' capability states from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path in self.data_dir.glob("*.json"):
            agent_name = path.stem
            self.load(agent_name)

    def stats(self) -> dict:
        """Registry statistics."""
        total_tokens = len(self.tokens)
        valid_tokens = sum(1 for t in self.tokens.values() if t.is_valid())
        all_actions: Set[CapabilityAction] = set()
        for t in self.tokens.values():
            if t.is_valid():
                all_actions.add(t.action)
        return {
            "total_tokens": total_tokens,
            "valid_tokens": valid_tokens,
            "revoked_tokens": total_tokens - valid_tokens,
            "agents_with_capabilities": len(self.agent_tokens),
            "unique_actions": len(all_actions),
            "action_types": [a.value for a in sorted(all_actions, key=lambda x: x.value)],
        }
