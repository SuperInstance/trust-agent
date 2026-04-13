#!/usr/bin/env python3
"""
Capability Middleware — Authorization Layer — Standalone Agent Module

Extracted from holodeck-studio capability_integration.py. Wires OCap capability
tokens into command handlers with dual-mode (OCap + ACL) authorization.
Zero external dependencies (stdlib only).

Design:
    - CommandActionMap: maps CLI commands to CapabilityActions
    - CapabilityMiddleware: OCap + ACL dual-mode authorization
    - CapabilityAudit: persistent JSONL audit trail
    - TrustBridge: connects TrustEngine to CapabilityRegistry
"""

import time
import json
import functools
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass

from capability_tokens import (
    CapabilityRegistry,
    CapabilityAction,
    BetaReputation,
    CapabilityToken,
    LEVEL_CAPABILITIES,
    EXERCISE_TRUST_THRESHOLD,
    ENDORSEMENT_TRUST_THRESHOLD,
    DELEGATION_TRUST_THRESHOLD,
    BASE_TRUST,
    TRUST_DECAY_ALERT_THRESHOLD,
)


# ── Singleton Registry ──────────────────────────────────────────────────────

_registry: Optional[CapabilityRegistry] = None


def get_registry(data_dir: str = "capability_data") -> CapabilityRegistry:
    """Get or create the singleton CapabilityRegistry instance."""
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry(data_dir=data_dir)
        _registry.load_all()
    return _registry


def reset_registry() -> None:
    """Reset the singleton registry (for testing)."""
    global _registry
    _registry = None


# ── CommandActionMap ────────────────────────────────────────────────────────

class CommandActionMap:
    """Maps CLI commands to their corresponding CapabilityAction."""

    _MAP: Dict[str, CapabilityAction] = {
        "build": CapabilityAction.BUILD_ROOM,
        "spawn": CapabilityAction.SUMMON_NPC,
        "write": CapabilityAction.CREATE_ITEM,
        "roomcmd": CapabilityAction.EDIT_ROOM,
        "backtest": CapabilityAction.CREATE_ADVENTURE,
        "review": CapabilityAction.REVIEW_AGENT,
        "ship": CapabilityAction.MANAGE_VESSEL,
        "setmotd": CapabilityAction.BROADCAST_FLEET,
        "hail": CapabilityAction.BROADCAST_FLEET,
        "cast": CapabilityAction.CREATE_SPELL,
        "install": CapabilityAction.CREATE_TOOL_ROOM,
        "budget": CapabilityAction.MANAGE_PERMISSIONS,
        "alert": CapabilityAction.GOVERN,
        "formality": CapabilityAction.GOVERN,
        "oversee": CapabilityAction.GOVERN,
        "shell": CapabilityAction.SHELL,
    }

    _ALIASES: Dict[str, str] = {
        "l": "look",
        "'": "say",
        "t": "tell",
        "g": "gossip",
        ":": "emote",
        "x": "examine",
        "?": "help",
        "exit": "quit",
        "move": "go",
    }

    @classmethod
    def get_action(cls, command: str) -> Optional[CapabilityAction]:
        """Get the CapabilityAction required by a command."""
        cmd = command.lower().strip()
        cmd = cls._ALIASES.get(cmd, cmd)
        return cls._MAP.get(cmd)

    @classmethod
    def is_gated(cls, command: str) -> bool:
        """Check if a command requires a capability token."""
        return cls.get_action(command) is not None

    @classmethod
    def all_gated_commands(cls) -> Dict[str, str]:
        """Return all gated commands with their action names."""
        return {cmd: action.value for cmd, action in cls._MAP.items()}

    @classmethod
    def commands_for_action(cls, action: CapabilityAction) -> List[str]:
        """Return all commands that require a given action."""
        return [cmd for cmd, a in cls._MAP.items() if a == action]


# ── CheckResult ─────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Result of a capability/permission check."""

    allowed: bool
    via: str  # "ocap", "acl", or "none"
    reason: str = ""
    agent: str = ""
    action: str = ""
    agent_level: int = 0
    required_level: int = 0
    token_id: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "via": self.via,
            "reason": self.reason,
            "agent": self.agent,
            "action": self.action,
            "agent_level": self.agent_level,
            "required_level": self.required_level,
            "token_id": self.token_id,
        }


# ── CapabilityMiddleware ────────────────────────────────────────────────────

class CapabilityMiddleware:
    """Middleware that wraps command handlers with capability token checks.

    Dual-mode authorization:
        1. OCap check: Does the agent hold a valid capability token?
        2. ACL fallback: Does the agent's permission_level permit this action?
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        permission_levels: Optional[Dict[str, int]] = None,
        mode: str = "dual",
    ) -> None:
        """Initialize with registry, optional ACL levels, and mode."""
        self.registry: CapabilityRegistry = registry
        self.permission_levels: Dict[str, int] = permission_levels or {}
        self.mode: str = mode
        self._audit_trail: List[dict] = []

    def check(self, agent_name: str, action: CapabilityAction) -> CheckResult:
        """Check if an agent can perform an action (OCap then ACL fallback)."""
        action_str = action.value
        agent_level = self.permission_levels.get(agent_name, 0)

        # OCap first
        if self.mode in ("dual", "ocap"):
            if self.registry.can_agent(agent_name, action):
                token_id = self._find_authorizing_token(agent_name, action)
                result = CheckResult(
                    allowed=True,
                    via="ocap",
                    reason=f"Agent holds valid capability token for {action_str}",
                    agent=agent_name,
                    action=action_str,
                    agent_level=agent_level,
                    token_id=token_id,
                )
                self._record(result)
                return result

        # ACL fallback
        if self.mode in ("dual", "acl"):
            required_level = self._acl_required_level(action)
            if required_level is not None and agent_level >= required_level:
                result = CheckResult(
                    allowed=True,
                    via="acl",
                    reason=f"Agent level {agent_level} >= required level {required_level}",
                    agent=agent_name,
                    action=action_str,
                    agent_level=agent_level,
                    required_level=required_level,
                )
                self._record(result)
                return result
            elif required_level is not None:
                result = CheckResult(
                    allowed=False,
                    via="none",
                    reason=f"Insufficient permissions. Level {required_level} required for {action_str}, agent is level {agent_level}",
                    agent=agent_name,
                    action=action_str,
                    agent_level=agent_level,
                    required_level=required_level,
                )
                self._record(result)
                return result

        # OCap-only mode: no token found
        if self.mode == "ocap":
            result = CheckResult(allowed=False, via="none", reason=f"No valid capability token for {action_str}", agent=agent_name, action=action_str, agent_level=agent_level)
            self._record(result)
            return result

        # Ungated action
        result = CheckResult(allowed=True, via="none", reason=f"Action {action_str} is ungated", agent=agent_name, action=action_str, agent_level=agent_level)
        self._record(result)
        return result

    def check_command(self, agent_name: str, command: str) -> CheckResult:
        """Check if an agent can execute a named command."""
        action = CommandActionMap.get_action(command)
        if action is None:
            return CheckResult(allowed=True, via="none", reason=f"Command '{command}' is ungated", agent=agent_name, action=command)
        return self.check(agent_name, action)

    def decorate(self, action: CapabilityAction) -> Callable:
        """Decorator factory for gating command handlers by capability."""
        def decorator(handler: Callable) -> Callable:
            @functools.wraps(handler)
            async def wrapper(self_cmd: Any, agent: Any, args: Any, **kwargs: Any) -> Any:
                agent_name = agent.name if hasattr(agent, "name") else str(agent)
                result = self.check(agent_name, action)
                if not result.allowed:
                    if hasattr(self_cmd, "send"):
                        await self_cmd.send(agent, f"[capability] {result.reason}")
                    return None
                return await handler(self_cmd, agent, args, **kwargs)
            return wrapper
        return decorator

    def _find_authorizing_token(self, agent_name: str, action: CapabilityAction) -> str:
        """Find the token_id that authorizes an agent's action."""
        token_ids = self.registry.agent_tokens.get(agent_name, set())
        for tid in token_ids:
            token = self.registry.tokens.get(tid)
            if token and token.can_exercise(action):
                return tid
        return ""

    def _acl_required_level(self, action: CapabilityAction) -> Optional[int]:
        """Get the minimum ACL level required for a CapabilityAction."""
        for level, actions in LEVEL_CAPABILITIES.items():
            if action in actions:
                return level
        return None

    def _record(self, result: CheckResult) -> None:
        """Record a check in the audit trail."""
        self._audit_trail.append({
            "timestamp": time.time(),
            "agent": result.agent,
            "action": result.action,
            "allowed": result.allowed,
            "via": result.via,
            "reason": result.reason,
        })

    @property
    def audit_trail(self) -> List[dict]:
        """Get the in-memory audit trail."""
        return list(self._audit_trail)

    def clear_audit(self) -> None:
        """Clear the in-memory audit trail."""
        self._audit_trail.clear()


# ── CapabilityAudit — Persistent JSONL Audit Trail ─────────────────────────

class CapabilityAudit:
    """Persistent audit trail for capability exercises and permission checks."""

    def __init__(self, filepath: str = "capability_audit.jsonl") -> None:
        """Initialize with a JSONL file path."""
        self.filepath: Path = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._in_memory: List[dict] = []
        self._load()

    def record(
        self,
        agent: str,
        action: str,
        allowed: bool,
        via: str = "",
        reason: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a capability check event."""
        entry: dict = {
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent": agent,
            "action": action,
            "allowed": allowed,
            "via": via,
            "reason": reason,
        }
        if metadata:
            entry["metadata"] = metadata

        self._in_memory.append(entry)

        try:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def recent_checks(
        self,
        agent: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Query the audit trail."""
        entries = self._in_memory
        if agent:
            entries = [e for e in entries if e["agent"] == agent]
        if action:
            entries = [e for e in entries if e["action"] == action]
        return list(reversed(entries[-limit:]))

    def denied_checks(self, agent: Optional[str] = None, limit: int = 50) -> List[dict]:
        """Get only denied checks."""
        return [e for e in self.recent_checks(agent=agent, limit=limit * 3) if not e["allowed"]][:limit]

    def stats(self) -> dict:
        """Audit trail statistics."""
        if not self._in_memory:
            return {"total": 0, "allowed": 0, "denied": 0, "by_agent": {}, "by_via": {}, "by_action": {}}
        total = len(self._in_memory)
        allowed = sum(1 for e in self._in_memory if e["allowed"])
        by_agent: Dict[str, int] = {}
        by_via: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        for e in self._in_memory:
            by_agent[e["agent"]] = by_agent.get(e["agent"], 0) + 1
            via = e.get("via", "none")
            by_via[via] = by_via.get(via, 0) + 1
            by_action[e["action"]] = by_action.get(e["action"], 0) + 1
        return {"total": total, "allowed": allowed, "denied": total - allowed, "by_agent": by_agent, "by_via": by_via, "by_action": by_action}

    def _load(self) -> None:
        """Load existing audit entries from JSONL file."""
        if not self.filepath.exists():
            return
        try:
            with open(self.filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._in_memory.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            pass

    def clear(self) -> None:
        """Clear in-memory trail and the file."""
        self._in_memory.clear()
        try:
            if self.filepath.exists():
                self.filepath.unlink()
        except OSError:
            pass


# ── TrustBridge — Connects TrustEngine to CapabilityRegistry ────────────────

class TrustBridge:
    """Bridges the TrustEngine to the CapabilityRegistry.

    Responsibilities:
        - Sets up trust_getter callback on the registry
        - Watches for trust changes and auto-suspends/restores capabilities
        - Auto-issues tokens on level-up via endowment
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        trust_engine: Any = None,
        permission_levels: Optional[Dict[str, int]] = None,
        audit: Optional[CapabilityAudit] = None,
    ) -> None:
        """Initialize the trust bridge."""
        self.registry: CapabilityRegistry = registry
        self.trust_engine: Any = trust_engine
        self.permission_levels: Dict[str, int] = permission_levels or {}
        self.audit: Optional[CapabilityAudit] = audit
        self._suspended_agents: Dict[str, float] = {}
        self._setup_trust_getter()

    def _setup_trust_getter(self) -> None:
        """Set up the trust_getter callback on the registry."""
        bridge = self

        def trust_getter(agent_name: str) -> float:
            if bridge.trust_engine:
                try:
                    return bridge.trust_engine.composite_trust(agent_name)
                except Exception:
                    pass
            return BASE_TRUST

        self.registry.set_trust_getter(trust_getter)

    def on_trust_change(self, agent: str, old_score: float, new_score: float) -> None:
        """Handle a trust score change — suspend or restore capabilities."""
        if agent in self._suspended_agents:
            restore_threshold = EXERCISE_TRUST_THRESHOLD + 0.05
            if new_score >= restore_threshold:
                del self._suspended_agents[agent]
                self._record_audit(agent, "restore", True, {"old_score": old_score, "new_score": new_score})
        else:
            if new_score < EXERCISE_TRUST_THRESHOLD:
                self._suspended_agents[agent] = time.time()
                self._record_audit(agent, "suspend", True, {
                    "old_score": old_score,
                    "new_score": new_score,
                    "suspended_tokens": len(self.registry.agent_tokens.get(agent, set())),
                })

    def endow_capabilities(
        self,
        agent: str,
        level: int,
        trust_score: Optional[float] = None,
    ) -> List[CapabilityToken]:
        """Endow an agent with capability tokens for their level."""
        old_level = self.permission_levels.get(agent, 0)
        tokens = self.registry.endow_on_level_up(agent, old_level, level, trust_score=trust_score)
        self.permission_levels[agent] = level

        if tokens:
            self._record_audit(agent, "endow", True, {
                "old_level": old_level,
                "new_level": level,
                "tokens_issued": len(tokens),
                "token_actions": [t.action.value for t in tokens],
            })

        return tokens

    def revoke_all_for_agent(self, agent: str, reason: str = "Trust revoked") -> None:
        """Revoke all capability tokens for an agent."""
        token_ids = list(self.registry.agent_tokens.get(agent, set()))
        for tid in token_ids:
            self.registry.revoke(tid, reason)
        self._record_audit(agent, "revoke_all", True, {"tokens_revoked": len(token_ids), "reason": reason})

    def is_suspended(self, agent: str) -> bool:
        """Check if an agent's capabilities are currently suspended."""
        return agent in self._suspended_agents

    def suspended_agents(self) -> List[str]:
        """Get list of currently suspended agent names."""
        return list(self._suspended_agents.keys())

    def _record_audit(self, agent: str, event: str, allowed: bool, metadata: dict) -> None:
        """Record a trust bridge event in the audit trail."""
        if self.audit:
            self.audit.record(
                agent=agent,
                action=f"trust_bridge:{event}",
                allowed=allowed,
                via="trust_bridge",
                reason=f"Trust bridge event: {event}",
                metadata=metadata,
            )
