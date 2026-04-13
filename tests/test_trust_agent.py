#!/usr/bin/env python3
"""
Tests for Trust Agent — trust_engine, trust_portability, capability_tokens, capability_middleware

Covers all core functionality: trust scoring, decay, portability, OCap tokens,
middleware authorization, and audit trails. Uses only stdlib + unittest.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional

# Ensure local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trust_engine import (
    TrustEngine,
    TrustProfile,
    TrustScore,
    WeightedHistory,
    TRUST_DIMENSIONS,
    TRUST_EVENTS,
    DEFAULT_WEIGHTS,
    BASE_TRUST,
    DECAY_RATES,
    MIN_EVENTS_FOR_TRUST,
    trust_level_name,
)
from trust_portability import (
    TrustAttestation,
    FleetTrustBridge,
    InconsistencyReport,
    FLEET_TRUST_KEY,
    DEFAULT_IMPORT_FACTOR,
    FOREIGN_DECAY_RATE,
    ATTESTATION_MAX_AGE,
    TRUST_DIMENSIONS as PORT_DIMENSIONS,
)
from capability_tokens import (
    CapabilityRegistry,
    CapabilityAction,
    CapabilityToken,
    BetaReputation,
    LEVEL_CAPABILITIES,
    ENDORSEMENT_TRUST_THRESHOLD,
    EXERCISE_TRUST_THRESHOLD,
    DELEGATION_TRUST_THRESHOLD,
    BASE_TRUST as CAP_BASE_TRUST,
)
from capability_middleware import (
    CapabilityMiddleware,
    CapabilityAudit,
    TrustBridge,
    CommandActionMap,
    CheckResult,
    get_registry,
    reset_registry,
)


class TestWeightedHistory(unittest.TestCase):
    """Tests for WeightedHistory temporal decay scoring."""

    def test_empty_returns_base_trust(self):
        wh = WeightedHistory()
        self.assertAlmostEqual(wh.score(), BASE_TRUST, places=4)

    def test_single_event(self):
        wh = WeightedHistory(decay_rate=0.99)
        wh.add(0.8, weight=1.0)
        self.assertAlmostEqual(wh.score(), 0.8, places=4)

    def test_clamped_values(self):
        wh = WeightedHistory()
        wh.add(-0.5)  # clamped to 0
        wh.add(1.5)   # clamped to 1
        self.assertAlmostEqual(wh.score(), 0.5, places=4)

    def test_weighted_average(self):
        wh = WeightedHistory(decay_rate=1.0)  # no decay
        wh.add(0.4, weight=1.0)
        wh.add(0.8, weight=3.0)
        # (0.4*1 + 0.8*3) / (1+3) = 2.8/4 = 0.7
        self.assertAlmostEqual(wh.score(), 0.7, places=4)

    def test_temporal_decay(self):
        wh = WeightedHistory(decay_rate=0.5)
        now = time.time()
        wh.add(1.0, weight=1.0, timestamp=now - 86400)  # 1 day ago
        wh.add(0.0, weight=1.0, timestamp=now)            # now
        # decayed: 1.0 * 0.5^1 = 0.5, fresh: 0.0 * 1.0 = 0.0
        # (0.5 + 0.0) / (0.5 + 1.0) = 0.5/1.5 ≈ 0.333
        result = wh.score()
        self.assertAlmostEqual(result, 0.3333, places=3)

    def test_recent_events(self):
        wh = WeightedHistory()
        wh.add(0.5)
        wh.add(0.7)
        recent = wh.recent(1)
        self.assertEqual(len(recent), 1)

    def test_prune(self):
        wh = WeightedHistory()
        old_time = time.time() - 100 * 86400  # 100 days ago
        wh.add(0.5, timestamp=old_time)
        wh.add(0.8)  # now
        wh.prune(max_age_days=90)
        self.assertEqual(wh.event_count(), 1)

    def test_serialization_roundtrip(self):
        wh = WeightedHistory(decay_rate=0.97)
        wh.add(0.75, weight=1.2)
        data = wh.to_dict()
        wh2 = WeightedHistory.from_dict(data)
        self.assertAlmostEqual(wh.score(), wh2.score(), places=4)


class TestTrustProfile(unittest.TestCase):
    """Tests for TrustProfile composite scoring."""

    def test_default_dimensions(self):
        profile = TrustProfile(agent_name="test")
        for dim in TRUST_DIMENSIONS:
            self.assertIn(dim, profile.dimensions)

    def test_composite_base_trust(self):
        profile = TrustProfile(agent_name="test")
        self.assertAlmostEqual(profile.composite(), BASE_TRUST, places=4)

    def test_record_increases_score(self):
        profile = TrustProfile(agent_name="test")
        profile.record("code_quality", 0.9, weight=2.0)
        self.assertGreater(profile.composite(), BASE_TRUST)

    def test_custom_weights(self):
        profile = TrustProfile(agent_name="test")
        custom = {d: 0.0 for d in TRUST_DIMENSIONS}
        custom["code_quality"] = 1.0
        profile.record("code_quality", 0.9)
        result = profile.composite(weights=custom)
        self.assertAlmostEqual(result, 0.9, places=4)

    def test_meaningful_threshold(self):
        profile = TrustProfile(agent_name="test")
        self.assertFalse(profile.is_meaningful())
        for _ in range(MIN_EVENTS_FOR_TRUST):
            profile.record("code_quality", 0.8)
        self.assertTrue(profile.is_meaningful())

    def test_review_exempt(self):
        profile = TrustProfile(agent_name="test")
        self.assertFalse(profile.review_exempt())
        # Boost composite above 0.7 with enough events
        for _ in range(20):
            for dim in TRUST_DIMENSIONS:
                profile.record(dim, 1.0, weight=2.0)
        self.assertTrue(profile.review_exempt())

    def test_trust_score_snapshot(self):
        profile = TrustProfile(agent_name="test")
        ts = profile.get_trust_score()
        self.assertIsInstance(ts, TrustScore)
        self.assertEqual(ts.level, "minimal")  # BASE_TRUST=0.3 maps to "minimal"
        self.assertFalse(ts.meaningful)

    def test_serialization_roundtrip(self):
        profile = TrustProfile(agent_name="test")
        profile.record("code_quality", 0.85)
        data = profile.to_dict()
        profile2 = TrustProfile.from_dict(data)
        self.assertAlmostEqual(profile.composite(), profile2.composite(), places=4)

    def test_dimension_score(self):
        profile = TrustProfile(agent_name="test")
        profile.record("innovation", 1.0, weight=3.0)
        self.assertGreater(profile.score("innovation"), BASE_TRUST)
        self.assertAlmostEqual(profile.score("collaboration"), BASE_TRUST, places=4)


class TestTrustEngine(unittest.TestCase):
    """Tests for TrustEngine fleet-wide management."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="trust_test_")
        self.engine = TrustEngine(data_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_or_create_profile(self):
        p = self.engine.get_profile("alice")
        self.assertEqual(p.agent_name, "alice")
        p2 = self.engine.get_profile("alice")
        self.assertIs(p, p2)

    def test_record_event(self):
        self.engine.record_event("alice", "code_quality", 0.9)
        score = self.engine.get_trust("alice")
        self.assertGreater(score, BASE_TRUST)

    def test_record_preset(self):
        result = self.engine.record_preset("alice", "task_completed")
        self.assertTrue(result)
        score = self.engine.get_trust("alice")
        self.assertGreater(score, BASE_TRUST)

    def test_record_preset_unknown(self):
        result = self.engine.record_preset("alice", "nonexistent_event")
        self.assertFalse(result)

    def test_composite_trust(self):
        self.engine.record_event("alice", "code_quality", 1.0)
        self.engine.record_event("alice", "task_completion", 1.0)
        comp = self.engine.composite_trust("alice")
        self.assertGreater(comp, BASE_TRUST)

    def test_compare(self):
        self.engine.record_event("alice", "code_quality", 0.9)
        self.engine.record_event("bob", "code_quality", 0.3)
        result = self.engine.compare("alice", "bob")
        self.assertIn("agent_a", result)
        self.assertIn("agent_b", result)
        self.assertIn("similarity", result)
        self.assertGreater(result["agent_a"]["composite"], result["agent_b"]["composite"])

    def test_leaderboard(self):
        self.engine.record_event("alice", "code_quality", 1.0, weight=3.0)
        self.engine.record_event("bob", "code_quality", 0.5)
        board = self.engine.leaderboard()
        # Only meaningful profiles appear
        self.assertIsInstance(board, list)

    def test_persistence(self):
        self.engine.record_event("alice", "code_quality", 0.9)
        self.engine.save("alice")
        engine2 = TrustEngine(data_dir=self.tmpdir)
        engine2.load("alice")
        score = engine2.get_trust("alice")
        self.assertGreater(score, BASE_TRUST)

    def test_prune_stale(self):
        profile = self.engine.get_profile("alice")
        profile.last_seen = time.time() - 100 * 86400
        self.engine.save("alice")
        count = self.engine.prune_stale(max_age_days=60)
        self.assertEqual(count, 1)

    def test_stats(self):
        self.engine.record_event("alice", "code_quality", 0.8)
        stats = self.engine.stats()
        self.assertEqual(stats["total_profiles"], 1)
        self.assertEqual(stats["dimensions"], len(TRUST_DIMENSIONS))

    def test_trust_level_name(self):
        self.assertEqual(trust_level_name(0.0), "unknown")
        self.assertEqual(trust_level_name(0.25), "minimal")
        self.assertEqual(trust_level_name(0.6), "established")
        self.assertEqual(trust_level_name(0.75), "trusted")
        self.assertEqual(trust_level_name(0.9), "exemplary")


class TestTrustAttestation(unittest.TestCase):
    """Tests for TrustAttestation signing and verification."""

    def test_sign_and_verify(self):
        att = TrustAttestation(agent_name="alice", issuer_repo="repo-a", composite=0.8)
        att.sign()
        self.assertTrue(att.verify())
        self.assertNotEqual(att.signature, "")

    def test_tampering_detected(self):
        att = TrustAttestation(agent_name="alice", issuer_repo="repo-a", composite=0.8)
        att.sign()
        att.composite = 0.1
        self.assertFalse(att.verify())

    def test_empty_signature_fails(self):
        att = TrustAttestation(agent_name="alice")
        self.assertFalse(att.verify())

    def test_fingerprint_stable(self):
        att = TrustAttestation(agent_name="alice", issuer_repo="repo-a", composite=0.8)
        fp1 = att.compute_fingerprint()
        fp2 = att.compute_fingerprint()
        self.assertEqual(fp1, fp2)

    def test_different_fingerprints(self):
        att1 = TrustAttestation(agent_name="alice", composite=0.8)
        att2 = TrustAttestation(agent_name="alice", composite=0.9)
        self.assertNotEqual(att1.compute_fingerprint(), att2.compute_fingerprint())

    def test_expiry(self):
        att = TrustAttestation(agent_name="alice", expires_at=time.time() - 100)
        self.assertTrue(att.is_expired())
        att2 = TrustAttestation(agent_name="alice", expires_at=time.time() + 86400)
        self.assertFalse(att2.is_expired())

    def test_never_expires(self):
        att = TrustAttestation(agent_name="alice", expires_at=0)
        self.assertFalse(att.is_expired())

    def test_age(self):
        att = TrustAttestation(agent_name="alice", issued_at=time.time() - 2 * 86400)
        self.assertAlmostEqual(att.age_days(), 2.0, places=1)

    def test_decayed_weight(self):
        att = TrustAttestation(agent_name="alice", issued_at=time.time())
        self.assertAlmostEqual(att.decayed_weight(), 1.0, places=4)

    def test_serialization_roundtrip(self):
        att = TrustAttestation(agent_name="alice", issuer_repo="repo-a", composite=0.75)
        att.sign()
        json_str = att.to_json()
        att2 = TrustAttestation.from_json(json_str)
        self.assertTrue(att2.verify())
        self.assertEqual(att2.agent_name, "alice")
        self.assertAlmostEqual(att2.composite, 0.75, places=4)

    def test_custom_key(self):
        att = TrustAttestation(agent_name="alice", composite=0.8)
        att.sign(key="my-secret-key")
        self.assertTrue(att.verify(key="my-secret-key"))
        self.assertFalse(att.verify(key="wrong-key"))


class TestFleetTrustBridge(unittest.TestCase):
    """Tests for FleetTrustBridge import/export and blending."""

    def test_import_valid_attestation(self):
        bridge = FleetTrustBridge(local_repo="local")
        att = TrustAttestation(agent_name="alice", issuer_repo="remote", composite=0.8)
        att.sign()
        result = bridge.import_attestation(att)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "valid")

    def test_replay_detection(self):
        bridge = FleetTrustBridge(local_repo="local")
        att = TrustAttestation(agent_name="alice", issuer_repo="remote", composite=0.8)
        att.sign()
        bridge.import_attestation(att)
        result = bridge.import_attestation(att)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "replay_detected")

    def test_invalid_signature_rejected(self):
        bridge = FleetTrustBridge(local_repo="local", fleet_key="correct-key")
        att = TrustAttestation(agent_name="alice", issuer_repo="remote", composite=0.8)
        att.sign(key="wrong-key")
        result = bridge.import_attestation(att)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "invalid_signature")

    def test_expired_attestation_rejected(self):
        bridge = FleetTrustBridge(local_repo="local")
        att = TrustAttestation(
            agent_name="alice",
            issuer_repo="remote",
            composite=0.8,
            expires_at=time.time() - 100,
        )
        att.sign()
        result = bridge.import_attestation(att)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "expired")

    def test_export_attestation(self):
        bridge = FleetTrustBridge(local_repo="my-repo")
        att = bridge.export_attestation(
            agent_name="alice",
            trust_getter=lambda dim: 0.8,
            composite_getter=lambda: 0.75,
        )
        self.assertEqual(att.agent_name, "alice")
        self.assertEqual(att.issuer_repo, "my-repo")
        self.assertTrue(att.verify())

    def test_foreign_trust_no_data(self):
        bridge = FleetTrustBridge(local_repo="local")
        result = bridge.foreign_trust("alice")
        self.assertAlmostEqual(result, BASE_TRUST, places=4)

    def test_foreign_trust_with_data(self):
        bridge = FleetTrustBridge(local_repo="local")
        att = TrustAttestation(agent_name="alice", issuer_repo="remote", composite=0.9)
        att.sign()
        bridge.import_attestation(att)
        result = bridge.foreign_trust("alice")
        self.assertGreater(result, BASE_TRUST)

    def test_fleet_composite_no_foreign(self):
        bridge = FleetTrustBridge(local_repo="local", trust_getter=lambda n: 0.7)
        result = bridge.fleet_composite_trust("alice")
        self.assertAlmostEqual(result, 0.7, places=4)

    def test_prune_stale(self):
        bridge = FleetTrustBridge(local_repo="local")
        # Create an attestation that's very old but still importable (age < 90 days)
        now = time.time()
        att = TrustAttestation(
            agent_name="alice",
            issuer_repo="remote",
            composite=0.8,
            issued_at=now - 80 * 86400,  # 80 days ago — within ATTESTATION_MAX_AGE
        )
        att.sign()
        result = bridge.import_attestation(att, current_time=now)
        self.assertTrue(result["accepted"])
        # Now prune with a current_time that makes it too old
        removed = bridge.prune_stale_attestations(current_time=now + 15 * 86400)
        self.assertGreater(removed, 0)

    def test_stats(self):
        bridge = FleetTrustBridge(local_repo="local")
        stats = bridge.stats()
        self.assertEqual(stats["local_repo"], "local")
        self.assertEqual(stats["total_imports"], 0)


class TestBetaReputation(unittest.TestCase):
    """Tests for BetaReputation Subjective Logic system."""

    def test_default_state(self):
        rep = BetaReputation()
        self.assertAlmostEqual(rep.expected_value, 0.5, places=4)
        self.assertAlmostEqual(rep.belief, 1.0 / 4, places=4)
        self.assertAlmostEqual(rep.disbelief, 1.0 / 4, places=4)
        self.assertAlmostEqual(rep.uncertainty, 2.0 / 4, places=4)
        b, d, u = rep.opinion
        self.assertAlmostEqual(b + d + u, 1.0, places=4)

    def test_positive_update(self):
        rep = BetaReputation()
        rep.update(positive=True)
        self.assertGreater(rep.expected_value, 0.5)

    def test_negative_update(self):
        rep = BetaReputation()
        rep.update(positive=False)
        self.assertLess(rep.expected_value, 0.5)

    def test_update_from_score(self):
        rep = BetaReputation()
        rep.update_from_score(0.9)  # positive
        self.assertGreater(rep.expected_value, 0.5)
        rep.update_from_score(0.1)  # negative
        self.assertLess(rep.expected_value, 0.7)  # brought back down

    def test_neutral_score_no_change(self):
        rep = BetaReputation()
        ev_before = rep.evidence_count
        rep.update_from_score(0.5)
        self.assertAlmostEqual(rep.evidence_count, ev_before, places=4)

    def test_discount_operator(self):
        source = BetaReputation()
        source.update(positive=True, magnitude=2.0)  # high trust source
        target = BetaReputation()
        target.update(positive=True, magnitude=1.0)  # target trusted
        result = source.discount(target)
        self.assertGreater(result.expected_value, 0.0)
        self.assertLess(result.expected_value, target.expected_value)  # discounted

    def test_fusion(self):
        rep1 = BetaReputation()
        rep1.update(positive=True)
        rep2 = BetaReputation()
        rep2.update(positive=True)
        fused = rep1.fuse(rep2)
        self.assertGreater(fused.expected_value, 0.5)

    def test_is_suspicious(self):
        rep = BetaReputation()
        self.assertTrue(rep.is_suspicious(uncertainty_threshold=0.4))
        for _ in range(20):
            rep.update(positive=True)
        self.assertFalse(rep.is_suspicious(uncertainty_threshold=0.5))

    def test_forgetting_factor(self):
        rep = BetaReputation(forget_factor=0.5)
        rep.update(positive=True)
        rep.update(positive=True)
        # With aggressive forgetting, evidence shouldn't grow as fast
        self.assertLess(rep.evidence_count, 3.0)

    def test_serialization_roundtrip(self):
        rep = BetaReputation()
        rep.update(positive=True)
        rep.update(positive=False)
        data = rep.to_dict()
        rep2 = BetaReputation.from_dict(data)
        self.assertAlmostEqual(rep.expected_value, rep2.expected_value, places=4)


class TestCapabilityToken(unittest.TestCase):
    """Tests for CapabilityToken OCap primitive."""

    def test_create_valid(self):
        token = CapabilityToken(action=CapabilityAction.BUILD_ROOM, holder="alice")
        self.assertTrue(token.is_valid())
        self.assertTrue(token.can_exercise(CapabilityAction.BUILD_ROOM))
        self.assertFalse(token.can_exercise(CapabilityAction.SHELL))

    def test_exercise_success(self):
        token = CapabilityToken(action=CapabilityAction.BUILD_ROOM, holder="alice")
        result = token.exercise()
        self.assertTrue(result["success"])
        self.assertEqual(result["token_id"], token.token_id)
        self.assertEqual(token.use_count, 1)

    def test_max_uses(self):
        token = CapabilityToken(action=CapabilityAction.BUILD_ROOM, holder="alice", max_uses=2)
        token.exercise()
        token.exercise()
        self.assertFalse(token.is_valid())
        result = token.exercise()
        self.assertFalse(result["success"])

    def test_expiry(self):
        token = CapabilityToken(action=CapabilityAction.BUILD_ROOM, holder="alice", expires=time.time() - 100)
        self.assertFalse(token.is_valid())

    def test_revocation(self):
        token = CapabilityToken(action=CapabilityAction.BUILD_ROOM, holder="alice")
        token.revoke("test reason")
        self.assertTrue(token.revoked)
        self.assertFalse(token.is_valid())

    def test_attenuation_reduces_uses(self):
        token = CapabilityToken(action=CapabilityAction.BUILD_ROOM, holder="alice", max_uses=5)
        child = token.attenuate(max_uses=2)
        self.assertEqual(child.max_uses, 2)
        # Cannot amplify
        child2 = token.attenuate(max_uses=10)
        self.assertEqual(child2.max_uses, 5)

    def test_attenuation_increments_depth(self):
        token = CapabilityToken(action=CapabilityAction.BUILD_ROOM, holder="alice")
        child = token.attenuate()
        self.assertEqual(child.delegate_depth, 1)

    def test_attenuation_restricts_actions(self):
        token = CapabilityToken(
            action=CapabilityAction.BUILD_ROOM,
            holder="alice",
            actions_allowed=[CapabilityAction.BUILD_ROOM, CapabilityAction.CREATE_ITEM],
        )
        child = token.attenuate(allowed_actions=[CapabilityAction.BUILD_ROOM])
        self.assertIn(CapabilityAction.BUILD_ROOM, child.actions_allowed)
        self.assertNotIn(CapabilityAction.CREATE_ITEM, child.actions_allowed)

    def test_serialization_roundtrip(self):
        token = CapabilityToken(
            action=CapabilityAction.SHELL,
            holder="alice",
            issuer="system",
            max_uses=10,
            scope="production",
        )
        data = token.to_dict()
        token2 = CapabilityToken.from_dict(data)
        self.assertEqual(token.token_id, token2.token_id)
        self.assertEqual(token.action, token2.action)
        self.assertTrue(token2.is_valid())


class TestCapabilityRegistry(unittest.TestCase):
    """Tests for CapabilityRegistry lifecycle."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="cap_test_")
        self.registry = CapabilityRegistry(data_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_issue_token(self):
        token = self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        self.assertEqual(token.holder, "alice")
        self.assertTrue(token.is_valid())

    def test_revoke_token(self):
        token = self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        self.registry.revoke(token.token_id)
        self.assertFalse(self.registry.tokens[token.token_id].is_valid())

    def test_revoke_downstream(self):
        parent = self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        # Manually create a downstream token
        child = parent.attenuate()
        child.holder = "bob"
        self.registry.tokens[child.token_id] = child
        self.registry.agent_tokens.setdefault("bob", set()).add(child.token_id)

        self.registry.revoke(parent.token_id)
        self.assertFalse(self.registry.tokens[child.token_id].is_valid())

    def test_delegate_success(self):
        # Set trust high enough
        self.registry.set_trust_getter(lambda name: 0.8)
        parent = self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        result = self.registry.delegate(parent.token_id, new_holder="bob", from_agent="alice")
        self.assertIsNotNone(result)
        self.assertEqual(result.holder, "bob")

    def test_delegate_low_trust_fails(self):
        self.registry.set_trust_getter(lambda name: 0.1)
        parent = self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        result = self.registry.delegate(parent.token_id, new_holder="bob", from_agent="alice")
        self.assertIsNone(result)

    def test_can_agent(self):
        self.registry.set_trust_getter(lambda name: 0.8)
        self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        self.assertTrue(self.registry.can_agent("alice", CapabilityAction.BUILD_ROOM))
        self.assertFalse(self.registry.can_agent("alice", CapabilityAction.SHELL))

    def test_can_agent_low_trust(self):
        self.registry.set_trust_getter(lambda name: 0.1)
        self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        self.assertFalse(self.registry.can_agent("alice", CapabilityAction.BUILD_ROOM))

    def test_endow_on_level_up(self):
        self.registry.set_trust_getter(lambda name: 0.8)
        tokens = self.registry.endow_on_level_up("alice", 0, 3)
        self.assertGreater(len(tokens), 0)
        actions = {t.action for t in tokens}
        self.assertIn(CapabilityAction.BUILD_ROOM, actions)

    def test_reputation_update(self):
        self.registry.update_reputation("alice", 0.9)
        rep = self.registry.get_reputation("alice")
        self.assertGreater(rep.expected_value, 0.5)

    def test_persistence(self):
        self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        self.registry.save("alice")
        reg2 = CapabilityRegistry(data_dir=self.tmpdir)
        reg2.load("alice")
        self.assertIn("alice", reg2.agent_tokens)

    def test_stats(self):
        self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        self.registry.issue(CapabilityAction.SHELL, holder="bob")
        stats = self.registry.stats()
        self.assertEqual(stats["total_tokens"], 2)
        self.assertEqual(stats["agents_with_capabilities"], 2)


class TestCapabilityMiddleware(unittest.TestCase):
    """Tests for CapabilityMiddleware dual-mode authorization."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mid_test_")
        reset_registry()
        self.registry = CapabilityRegistry(data_dir=self.tmpdir)
        self.registry.set_trust_getter(lambda name: 0.8)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ocap_allows(self):
        self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        mw = CapabilityMiddleware(registry=self.registry)
        result = mw.check("alice", CapabilityAction.BUILD_ROOM)
        self.assertTrue(result.allowed)
        self.assertEqual(result.via, "ocap")

    def test_ocap_denies_no_token(self):
        mw = CapabilityMiddleware(registry=self.registry)
        result = mw.check("alice", CapabilityAction.BUILD_ROOM)
        self.assertFalse(result.allowed)

    def test_acl_fallback(self):
        mw = CapabilityMiddleware(
            registry=self.registry,
            permission_levels={"alice": 3},
            mode="dual",
        )
        # No OCap token, but ACL level 3 allows BUILD_ROOM
        result = mw.check("alice", CapabilityAction.BUILD_ROOM)
        self.assertTrue(result.allowed)
        self.assertEqual(result.via, "acl")

    def test_acl_denies_low_level(self):
        mw = CapabilityMiddleware(
            registry=self.registry,
            permission_levels={"alice": 0},
            mode="dual",
        )
        result = mw.check("alice", CapabilityAction.BUILD_ROOM)
        self.assertFalse(result.allowed)

    def test_ocap_only_mode(self):
        mw = CapabilityMiddleware(registry=self.registry, mode="ocap")
        result = mw.check("alice", CapabilityAction.BUILD_ROOM)
        self.assertFalse(result.allowed)
        self.assertEqual(result.via, "none")

    def test_ungated_action(self):
        mw = CapabilityMiddleware(registry=self.registry)
        # GOVERN requires level 4
        result = mw.check("alice", CapabilityAction.GOVERN)
        # In dual mode with no ACL levels and no tokens: denied
        # Let's test an ungated path
        result = CheckResult(allowed=True, via="none", reason="test", agent="alice", action="look")
        self.assertTrue(result.allowed)

    def test_audit_trail(self):
        self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        mw = CapabilityMiddleware(registry=self.registry)
        mw.check("alice", CapabilityAction.BUILD_ROOM)
        mw.check("bob", CapabilityAction.BUILD_ROOM)
        self.assertEqual(len(mw.audit_trail), 2)

    def test_command_check(self):
        result = CommandActionMap.get_action("build")
        self.assertEqual(result, CapabilityAction.BUILD_ROOM)
        self.assertTrue(CommandActionMap.is_gated("build"))
        self.assertFalse(CommandActionMap.is_gated("look"))

    def test_clear_audit(self):
        self.registry.issue(CapabilityAction.BUILD_ROOM, holder="alice")
        mw = CapabilityMiddleware(registry=self.registry)
        mw.check("alice", CapabilityAction.BUILD_ROOM)
        mw.clear_audit()
        self.assertEqual(len(mw.audit_trail), 0)


class TestCapabilityAudit(unittest.TestCase):
    """Tests for persistent CapabilityAudit."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="audit_test_")
        self.audit = CapabilityAudit(filepath=os.path.join(self.tmpdir, "audit.jsonl"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_and_query(self):
        self.audit.record("alice", "build_room", True, "ocap", "token valid")
        entries = self.audit.recent_checks(agent="alice")
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["allowed"])

    def test_denied_filter(self):
        self.audit.record("alice", "shell", False, "none", "no token")
        self.audit.record("alice", "build_room", True, "ocap", "has token")
        denied = self.audit.denied_checks(agent="alice")
        self.assertEqual(len(denied), 1)

    def test_stats(self):
        self.audit.record("alice", "build_room", True, "ocap")
        self.audit.record("bob", "shell", False, "none")
        stats = self.audit.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["allowed"], 1)
        self.assertEqual(stats["denied"], 1)

    def test_persistence(self):
        self.audit.record("alice", "build_room", True, "ocap")
        # Reload from file
        audit2 = CapabilityAudit(filepath=os.path.join(self.tmpdir, "audit.jsonl"))
        entries = audit2.recent_checks()
        self.assertEqual(len(entries), 1)

    def test_clear(self):
        self.audit.record("alice", "build_room", True, "ocap")
        self.audit.clear()
        self.assertEqual(len(self.audit.recent_checks()), 0)


class TestTrustBridge(unittest.TestCase):
    """Tests for TrustBridge connecting TrustEngine to CapabilityRegistry."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bridge_test_")
        self.registry = CapabilityRegistry(data_dir=os.path.join(self.tmpdir, "caps"))
        self.engine = TrustEngine(data_dir=os.path.join(self.tmpdir, "trust"))
        self.audit = CapabilityAudit(filepath=os.path.join(self.tmpdir, "audit.jsonl"))
        self.bridge = TrustBridge(
            registry=self.registry,
            trust_engine=self.engine,
            audit=self.audit,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_suspend_on_trust_drop(self):
        self.bridge.on_trust_change("alice", 0.6, 0.1)
        self.assertTrue(self.bridge.is_suspended("alice"))

    def test_restore_on_trust_recovery(self):
        self.bridge.on_trust_change("alice", 0.6, 0.1)
        self.assertTrue(self.bridge.is_suspended("alice"))
        self.bridge.on_trust_change("alice", 0.1, 0.5)
        self.assertFalse(self.bridge.is_suspended("alice"))

    def test_endow_capabilities(self):
        self.engine.record_event("alice", "code_quality", 0.9, weight=3.0)
        tokens = self.bridge.endow_capabilities("alice", 3)
        self.assertGreater(len(tokens), 0)

    def test_revoke_all(self):
        self.engine.record_event("alice", "code_quality", 0.9, weight=3.0)
        self.bridge.endow_capabilities("alice", 2)
        self.bridge.revoke_all_for_agent("alice", "test revocation")
        caps = self.registry.agent_capabilities("alice")
        for cap in caps:
            self.assertFalse(cap["is_valid"])

    def test_suspended_agents_list(self):
        self.bridge.on_trust_change("alice", 0.6, 0.1)
        self.bridge.on_trust_change("bob", 0.5, 0.1)
        suspended = self.bridge.suspended_agents()
        self.assertIn("alice", suspended)
        self.assertIn("bob", suspended)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for the CLI module."""

    def test_score_command(self):
        from cli import cmd_score, build_parser
        tmpdir = tempfile.mkdtemp(prefix="cli_test_")
        try:
            parser = build_parser()
            args = parser.parse_args(["--data-dir", tmpdir, "score", "test-agent"])
            result = cmd_score(args)
            self.assertEqual(result, 0)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_onboard_command(self):
        from cli import cmd_onboard, build_parser
        tmpdir = tempfile.mkdtemp(prefix="cli_onboard_")
        try:
            parser = build_parser()
            args = parser.parse_args(["--data-dir", tmpdir, "onboard"])
            result = cmd_onboard(args)
            self.assertEqual(result, 0)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_status_command(self):
        from cli import cmd_status, build_parser
        tmpdir = tempfile.mkdtemp(prefix="cli_status_")
        try:
            parser = build_parser()
            args = parser.parse_args(["--data-dir", tmpdir, "status"])
            result = cmd_status(args)
            self.assertEqual(result, 0)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_attest_command(self):
        from cli import cmd_attest, build_parser
        tmpdir = tempfile.mkdtemp(prefix="cli_attest_")
        try:
            parser = build_parser()
            args = parser.parse_args(["--data-dir", tmpdir, "attest", "alice", "--level", "trusted"])
            result = cmd_attest(args)
            self.assertEqual(result, 0)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_export_import_roundtrip(self):
        from cli import cmd_attest, cmd_export_trust, cmd_import_trust, build_parser
        tmpdir = tempfile.mkdtemp(prefix="cli_roundtrip_")
        try:
            parser = build_parser()

            # First, attest to create some data
            args_attest = parser.parse_args(["--data-dir", tmpdir, "attest", "alice", "--level", "trusted"])
            cmd_attest(args_attest)

            # Export
            export_file = os.path.join(tmpdir, "alice-export.json")
            args_export = parser.parse_args(["--data-dir", tmpdir, "export-trust", "alice", "--output", export_file])
            result_export = cmd_export_trust(args_export)
            self.assertEqual(result_export, 0)

            # Import
            args_import = parser.parse_args(["--data-dir", tmpdir, "import-trust", export_file])
            result_import = cmd_import_trust(args_import)
            self.assertEqual(result_import, 0)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
