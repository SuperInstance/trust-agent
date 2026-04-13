#!/usr/bin/env python3
"""
Trust Agent — CLI Interface

Standalone CLI for managing trust scores, capability tokens, attestations,
and audit trails. Zero external dependencies (stdlib only).

Usage:
    python cli.py score <agent_id>
    python cli.py attest <agent_id> --level <level>
    python cli.py token create --capability <cap> --holder <agent> [--delegatable]
    python cli.py token delegate <token_id> --to <agent>
    python cli.py token revoke <token_id>
    python cli.py audit [--agent <id>]
    python cli.py export-trust <agent_id> [--output <file>]
    python cli.py import-trust <file>
    python cli.py onboard
    python cli.py status
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trust_engine import TrustEngine, TRUST_EVENTS, BASE_TRUST
from trust_portability import TrustAttestation, FleetTrustBridge, FLEET_TRUST_KEY
from capability_tokens import (
    CapabilityRegistry,
    CapabilityAction,
    LEVEL_CAPABILITIES,
    BASE_TRUST as CAP_BASE_TRUST,
)
from capability_middleware import (
    CapabilityMiddleware,
    CapabilityAudit,
    TrustBridge,
    get_registry,
    reset_registry,
)


# ── Data Directory ──────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _ensure_data_dir(data_dir: str) -> str:
    """Ensure the data directory exists and return its absolute path."""
    path = os.path.abspath(data_dir)
    os.makedirs(path, exist_ok=True)
    return path


# ── Helpers ─────────────────────────────────────────────────────────────────

def _print_json(data: object) -> None:
    """Pretty-print a JSON object."""
    print(json.dumps(data, indent=2, default=str))


def _load_engine(data_dir: str) -> TrustEngine:
    """Load or create a TrustEngine."""
    engine = TrustEngine(data_dir=os.path.join(data_dir, "trust"))
    engine.load_all()
    return engine


def _load_registry(data_dir: str) -> CapabilityRegistry:
    """Load or create a CapabilityRegistry."""
    reset_registry()
    registry = get_registry(data_dir=os.path.join(data_dir, "capabilities"))
    return registry


def _resolve_action(cap_name: str) -> Optional[CapabilityAction]:
    """Resolve a capability name string to a CapabilityAction enum."""
    cap_upper = cap_name.upper().replace("-", "_")
    try:
        return CapabilityAction[cap_upper]
    except KeyError:
        # Try value match
        for action in CapabilityAction:
            if action.value == cap_name.lower().replace("-", "_"):
                return action
        return None


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_score(args: argparse.Namespace) -> int:
    """Show trust score for an agent."""
    engine = _load_engine(args.data_dir)
    ts = engine.get_trust_score(args.agent_id)
    profile = engine.get_profile(args.agent_id)

    result = {
        "agent": args.agent_id,
        "composite": round(ts.composite, 4),
        "level": ts.level,
        "meaningful": ts.meaningful,
        "review_exempt": ts.review_exempt,
        "total_events": ts.total_events,
        "dimensions": {k: round(v, 4) for k, v in ts.dimensions.items()},
    }

    # Include recent events
    recent_events: list = []
    for dim in profile.dimensions:
        recent_events.extend(profile.dimensions[dim].recent(3))
    if recent_events:
        recent_events.sort(key=lambda e: e["timestamp"], reverse=True)
        result["recent_events"] = recent_events[:10]

    _print_json(result)
    return 0


def cmd_attest(args: argparse.Namespace) -> int:
    """Create a trust attestation for an agent."""
    engine = _load_engine(args.data_dir)
    bridge = FleetTrustBridge(
        local_repo=args.issuer or "trust-agent",
        fleet_key=args.key or FLEET_TRUST_KEY,
        trust_getter=lambda name: engine.composite_trust(name),
    )

    profile = engine.get_profile(args.agent_id)

    # If level specified, record some events to establish that level
    if args.level:
        level_map = {
            "minimal": 0.25,
            "established": 0.55,
            "trusted": 0.75,
            "exemplary": 0.9,
        }
        target = level_map.get(args.level.lower())
        if target:
            for dim in profile.dimensions:
                profile.record(dim, target, weight=1.5)

    # Create and sign attestation
    attestation = bridge.export_attestation(
        agent_name=args.agent_id,
        trust_getter=lambda dim: profile.score(dim),
        composite_getter=lambda: profile.composite(),
        event_count_getter=lambda: sum(h.event_count() for h in profile.dimensions.values()),
        meaningful_getter=lambda: profile.is_meaningful(),
    )

    # Save to file if output specified
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(attestation.to_json())
        print(f"Attestation saved to {args.output}")

    # Save engine state
    engine.save(args.agent_id)

    _print_json(attestation.to_dict())
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    """Handle token subcommands."""
    if args.token_command == "create":
        return _token_create(args)
    elif args.token_command == "delegate":
        return _token_delegate(args)
    elif args.token_command == "revoke":
        return _token_revoke(args)
    elif args.token_command == "list":
        return _token_list(args)
    else:
        print(f"Unknown token command: {args.token_command}", file=sys.stderr)
        return 1


def _token_create(args: argparse.Namespace) -> int:
    """Create a new capability token."""
    registry = _load_registry(args.data_dir)

    action = _resolve_action(args.capability)
    if not action:
        print(f"Unknown capability: {args.capability}", file=sys.stderr)
        print(f"Available: {', '.join(a.value for a in CapabilityAction)}", file=sys.stderr)
        return 1

    token = registry.issue(
        action=action,
        holder=args.holder,
        issuer=args.issuer or "cli",
        scope=args.scope or "",
        max_uses=args.max_uses or 0,
    )

    registry.save(args.holder)
    _print_json(token.to_dict())
    return 0


def _token_delegate(args: argparse.Namespace) -> int:
    """Delegate a capability token to another agent."""
    registry = _load_registry(args.data_dir)

    new_token = registry.delegate(
        token_id=args.token_id,
        new_holder=args.to_agent,
        from_agent=args.from_agent or "cli",
        max_uses=args.max_uses or 0,
    )

    if not new_token:
        print("Delegation failed: token not found, invalid, or trust thresholds not met.", file=sys.stderr)
        return 1

    registry.save(args.to_agent)
    _print_json(new_token.to_dict())
    return 0


def _token_revoke(args: argparse.Namespace) -> int:
    """Revoke a capability token."""
    registry = _load_registry(args.data_dir)

    if args.token_id not in registry.tokens:
        print(f"Token not found: {args.token_id}", file=sys.stderr)
        return 1

    token = registry.tokens[args.token_id]
    holder = token.holder
    registry.revoke(args.token_id, reason=args.reason or "Revoked via CLI")

    if holder:
        registry.save(holder)

    print(f"Token {args.token_id} revoked (downstream tokens also revoked).")
    return 0


def _token_list(args: argparse.Namespace) -> int:
    """List all tokens or tokens for a specific agent."""
    registry = _load_registry(args.data_dir)

    if args.agent:
        summary = registry.agent_summary(args.agent)
        _print_json(summary)
    else:
        _print_json(registry.stats())
        # Show per-agent summary
        for agent_name in sorted(registry.agent_tokens.keys()):
            caps = registry.agent_capabilities(agent_name)
            valid = [c for c in caps if c["is_valid"]]
            print(f"  {agent_name}: {len(valid)} valid, {len(caps)} total")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Show capability audit trail."""
    audit = CapabilityAudit(filepath=os.path.join(args.data_dir, "audit.jsonl"))

    if args.agent:
        entries = audit.recent_checks(agent=args.agent, limit=args.limit or 100)
    else:
        entries = audit.recent_checks(limit=args.limit or 100)

    _print_json(entries)

    # Print summary stats
    stats = audit.stats()
    print(f"\n--- Audit Stats ---")
    print(f"Total: {stats['total']}  Allowed: {stats['allowed']}  Denied: {stats['denied']}")
    if stats.get("by_agent"):
        print(f"By agent: {json.dumps(stats['by_agent'])}")
    if stats.get("by_via"):
        print(f"By auth path: {json.dumps(stats['by_via'])}")

    return 0


def cmd_export_trust(args: argparse.Namespace) -> int:
    """Export a trust profile as a signed attestation file."""
    engine = _load_engine(args.data_dir)
    bridge = FleetTrustBridge(
        local_repo=args.issuer or "trust-agent",
        fleet_key=args.key or FLEET_TRUST_KEY,
    )

    profile = engine.get_profile(args.agent_id)
    attestation = bridge.export_attestation(
        agent_name=args.agent_id,
        trust_getter=lambda dim: profile.score(dim),
        composite_getter=lambda: profile.composite(),
        event_count_getter=lambda: sum(h.event_count() for h in profile.dimensions.values()),
        meaningful_getter=lambda: profile.is_meaningful(),
    )

    output = args.output or f"{args.agent_id}-trust.json"
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(attestation.to_json())
    print(f"Trust profile exported to {output}")
    _print_json(attestation.to_dict())
    return 0


def cmd_import_trust(args: argparse.Namespace) -> int:
    """Import a trust attestation from a file."""
    bridge = FleetTrustBridge(
        local_repo="trust-agent",
        fleet_key=args.key or FLEET_TRUST_KEY,
    )

    try:
        attestation = TrustAttestation.from_json(Path(args.file).read_text())
    except Exception as e:
        print(f"Failed to read attestation file: {e}", file=sys.stderr)
        return 1

    result = bridge.import_attestation(attestation)
    _print_json(result)

    if result["accepted"]:
        print(f"\nAttestation for '{result['agent_name']}' accepted (issuer: {result['issuer_repo']}).")
    else:
        print(f"\nAttestation rejected: {result['reason']}", file=sys.stderr)

    return 0 if result["accepted"] else 1


def cmd_onboard(args: argparse.Namespace) -> int:
    """Set up the trust agent — initialize data directories and config."""
    data_dir = _ensure_data_dir(args.data_dir)

    # Create subdirectories
    os.makedirs(os.path.join(data_dir, "trust"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "capabilities"), exist_ok=True)

    # Write config
    config = {
        "agent_name": args.name or "trust-agent",
        "data_dir": data_dir,
        "fleet_key": args.key or FLEET_TRUST_KEY,
        "version": "1.0.0",
    }
    config_path = os.path.join(data_dir, "config.json")
    Path(config_path).write_text(json.dumps(config, indent=2))

    # Initialize engine and registry
    engine = _load_engine(data_dir)
    registry = _load_registry(data_dir)

    print(f"Trust agent onboarded successfully.")
    print(f"  Name:       {config['agent_name']}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Engine:     {engine.stats()}")
    print(f"  Registry:   {registry.stats()}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show agent status."""
    engine = _load_engine(args.data_dir)
    registry = _load_registry(args.data_dir)

    config_path = os.path.join(args.data_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        config = json.loads(Path(config_path).read_text())

    # Check for bridge data
    bridge_stats = {}
    bridge_path = os.path.join(args.data_dir, "bridge.json")
    if os.path.exists(bridge_path):
        bridge_stats = json.loads(Path(bridge_path).read_text()).get("stats", {})

    result = {
        "agent": config.get("agent_name", "trust-agent"),
        "version": config.get("version", "unknown"),
        "data_dir": os.path.abspath(args.data_dir),
        "trust_engine": engine.stats(),
        "capability_registry": registry.stats(),
        "fleet_bridge": bridge_stats,
    }

    _print_json(result)
    return 0


# ── Argument Parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="trust-agent",
        description="Standalone trust engine and OCap capability token CLI agent",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Data directory path")
    parser.add_argument("--key", default=None, help="Fleet trust key for attestation signing")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # score
    p_score = subparsers.add_parser("score", help="Show trust score for an agent")
    p_score.add_argument("agent_id", help="Agent identifier")
    p_score.set_defaults(func=cmd_score)

    # attest
    p_attest = subparsers.add_parser("attest", help="Create a trust attestation")
    p_attest.add_argument("agent_id", help="Agent identifier")
    p_attest.add_argument("--level", default=None, choices=["minimal", "established", "trusted", "exemplary"], help="Set trust level")
    p_attest.add_argument("--issuer", default=None, help="Issuer repo name")
    p_attest.add_argument("--output", "-o", default=None, help="Output file path")
    p_attest.set_defaults(func=cmd_attest)

    # token (sub-subcommand)
    p_token = subparsers.add_parser("token", help="Manage capability tokens")
    p_token.add_argument("token_command", choices=["create", "delegate", "revoke", "list"], help="Token action")

    # token create args
    p_token.add_argument("--capability", help="Capability action name (e.g., build_room)")
    p_token.add_argument("--holder", help="Token holder agent name")
    p_token.add_argument("--issuer", default=None, help="Token issuer")
    p_token.add_argument("--scope", default=None, help="Scope restriction")
    p_token.add_argument("--max-uses", type=int, default=None, help="Maximum uses (0=unlimited)")
    p_token.add_argument("--delegatable", action="store_true", help="Token is delegatable")

    # token delegate args
    p_token.add_argument("--token-id", help="Token ID to delegate")
    p_token.add_argument("--to", dest="to_agent", help="Delegate to agent")
    p_token.add_argument("--from", dest="from_agent", default=None, help="Delegating agent")

    # token revoke args
    p_token.add_argument("--reason", default=None, help="Revocation reason")

    # token list args
    p_token.add_argument("--agent", default=None, help="Agent to list tokens for")
    p_token.add_argument("--limit", type=int, default=None, help="Max entries")

    p_token.set_defaults(func=cmd_token)

    # audit
    p_audit = subparsers.add_parser("audit", help="Show capability audit trail")
    p_audit.add_argument("--agent", default=None, help="Filter by agent")
    p_audit.add_argument("--limit", type=int, default=None, help="Max entries")
    p_audit.set_defaults(func=cmd_audit)

    # export-trust
    p_export = subparsers.add_parser("export-trust", help="Export trust profile as attestation")
    p_export.add_argument("agent_id", help="Agent identifier")
    p_export.add_argument("--output", "-o", default=None, help="Output file path")
    p_export.add_argument("--issuer", default=None, help="Issuer repo name")
    p_export.set_defaults(func=cmd_export_trust)

    # import-trust
    p_import = subparsers.add_parser("import-trust", help="Import a trust attestation")
    p_import.add_argument("file", help="Attestation JSON file path")
    p_import.set_defaults(func=cmd_import_trust)

    # onboard
    p_onboard = subparsers.add_parser("onboard", help="Set up the trust agent")
    p_onboard.add_argument("--name", default=None, help="Agent name")
    p_onboard.set_defaults(func=cmd_onboard)

    # status
    p_status = subparsers.add_parser("status", help="Show agent status")
    p_status.set_defaults(func=cmd_status)

    return parser


# ── Main ────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    return func(args)


if __name__ == "__main__":
    sys.exit(main())
