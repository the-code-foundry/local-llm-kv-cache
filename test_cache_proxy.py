import json
import os
import runpy
import signal
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import cache_proxy
from cache_core import PrefixManifest, cache_filename, cache_key, manifest_filename
from cache_proxy import (
    MAX_METADATA_BYTES,
    CompletionMetadata,
    ForwardResult,
    LlamaCacheProxy,
    ProxyHandler,
    SnapshotPlan,
    SlotState,
    _anonymous_session_id,
    _body_session_id,
    _completion_metadata,
    _env_bool,
    _has_media,
    _metadata_from_object,
    _session_id,
    _without_proxy_affinity_fields,
)


class FakeLlamaHandler(BaseHTTPRequestHandler):
    cache_dir = None
    restored = []
    saved = []
    restore_failed = False
    restore_tokens = 10
    slot_tokens = 0

    def do_GET(self):
        if self.path == "/slots":
            self._json([{"id": 0, "is_processing": False, "n_prompt_tokens": self.slot_tokens}])
            return
        self._json({"status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if "action=restore" in self.path:
            self.restored.append(payload["filename"])
            if self.restore_failed:
                self._json({"error": "incompatible snapshot"}, status=500)
                return
            type(self).slot_tokens = self.restore_tokens
            self._json({"n_restored": self.restore_tokens})
            return
        if "action=save" in self.path:
            filename = payload["filename"]
            self.saved.append(filename)
            Path(self.cache_dir, filename).write_bytes(b"snapshot")
            type(self).slot_tokens = 10
            self._json({"n_saved": 10})
            return
        self._json({"status": "ok"})

    def _json(self, value, status=200):
        raw = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args):
        pass


class UnavailableProxy:
    require_session_id = False

    @contextmanager
    def foreground_operation(self):
        yield

    @contextmanager
    def session_operation(self, _session_id):
        yield

    def begin_request(self, body, session_id, controls=None):
        return self.prepare(body, session_id)

    def prepare(self, *_args):
        raise ConnectionRefusedError("llama is unavailable")


class CapturingProxy:
    require_session_id = False

    def __init__(self):
        self.session_ids = []
        self.forwarded = []
        self.uncached_reasons = []

    @contextmanager
    def foreground_operation(self):
        yield

    @contextmanager
    def session_operation(self, _session_id):
        yield

    def prepare(self, body, session_id):
        self.session_ids.append(session_id)
        return body, None

    def begin_request(self, body, session_id, controls=None):
        return self.prepare(body, session_id)

    def prepare_uncached(self, reason):
        self.uncached_reasons.append(reason)

    def forward(self, handler, *_args):
        self.forwarded.append(_args)
        raw = b"{}"
        setattr(handler, "_proxy_response_started", True)
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        if _args[0] != "HEAD":
            handler.wfile.write(raw)
        return ForwardResult(200)

    def finish(self, *_args, **_kwargs):
        pass

    def complete_request(self, *args, **kwargs):
        self.finish(*args, **kwargs)

    def cancel_request(self, _plan):
        pass


class FailingForwardProxy(CapturingProxy):
    def forward(self, *_args):
        raise OSError("upstream unavailable")


class FinishFailingProxy(CapturingProxy):
    def finish(self, *_args, **_kwargs):
        raise RuntimeError("snapshot finalization failed")


class FakeResponse:
    def __init__(self, status=200, raw=b"{}", headers=None, chunks=None):
        self.status = status
        self.reason = "OK"
        self._raw = raw
        self._headers = headers or []
        self._chunks = list(chunks) if chunks is not None else None
        self.read_calls = 0
        self.read1_calls = 0

    def getheaders(self):
        return self._headers

    def read(self, _size=None):
        self.read_calls += 1
        if self._chunks is None:
            return self._raw
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def read1(self, _size=None):
        self.read1_calls += 1
        if self._chunks is None:
            raw, self._raw = self._raw, b""
            return raw
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class FakeConnection:
    def __init__(self, _host, _port, timeout=None, response=None):
        self.timeout = timeout
        self.response = response or FakeResponse()
        self.requests = []
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class RecordingWriter:
    def __init__(self, broken=False):
        self.broken = broken
        self.closed = False
        self.data = []

    def write(self, data):
        if self.broken:
            raise BrokenPipeError("client disconnected")
        self.data.append(data)
        return len(data)

    def flush(self):
        pass


class RecordingHandler:
    def __init__(self, headers=None, broken=False):
        self.headers = headers or {}
        self.wfile = RecordingWriter(broken=broken)
        self.responses = []
        self.sent_headers = []
        self.errors = []
        self.ended = False

    def send_response(self, status, reason=None):
        self.responses.append((status, reason))

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        self.ended = True

    def send_error(self, status, message):
        self.errors.append((status, message))


class CacheProxyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        FakeLlamaHandler.cache_dir = self.tempdir.name
        FakeLlamaHandler.restored = []
        FakeLlamaHandler.saved = []
        FakeLlamaHandler.restore_failed = False
        FakeLlamaHandler.restore_tokens = 10
        FakeLlamaHandler.slot_tokens = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=1,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        self.body = {
            "model": "qwen3.8:27b",
            "messages": [
                {"role": "system", "content": "project rules"},
                {"role": "user", "content": "request"},
            ],
            "tools": [],
        }

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tempdir.cleanup()

    def _write_shared(self, name, tokens, namespace="default", scope="default"):
        snapshot = Path(self.tempdir.name, name)
        snapshot.write_bytes(b"snapshot")
        manifest = PrefixManifest(namespace, scope, name, tuple(tokens))
        Path(self.tempdir.name, manifest_filename(name)).write_text(manifest.to_json())
        return snapshot

    def test_shared_prefixes_can_be_disabled_for_session_snapshot_only_backends(self):
        session_only = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            enable_prefix_seeding=False,
            enable_shared_prefixes=False,
        )
        self._write_shared("local-llm-prefix-shared.bin", [1, 2, 3])
        legacy = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        legacy.write_bytes(b"legacy")
        session_only._render_tokens = Mock(side_effect=AssertionError("prompt APIs must not be called"))
        session_only._restore = Mock(return_value=3)

        request, plan = session_only.prepare(self.body, "new-session")

        self.assertNotIn("id_slot", request)
        self.assertIsNone(plan.slot_id)
        self.assertFalse(plan.prefix_was_present)
        self.assertIsNone(plan.stable_prefix_seed_tokens)
        session_only._render_tokens.assert_not_called()
        session_only._restore.assert_not_called()

    def test_prepare_seeds_verified_lcp_instead_of_restoring_divergent_snapshot(self):
        short = self._write_shared("local-llm-prefix-short.bin", [1, 2])
        longest = self._write_shared("local-llm-prefix-long.bin", [1, 2, 3, 4])
        self.proxy.minimum_shared_prefix_tokens = 2
        calls = []

        def api(_method, path, payload):
            calls.append((path, payload))
            if path == "/apply-template":
                return {"prompt": "rendered-private-prompt"}
            if path == "/tokenize":
                return {"tokens": [1, 2, 3, 9]}
            self.fail(path)

        self.proxy._json_request = Mock(side_effect=api)
        self.proxy._restore = Mock()

        with patch.object(cache_proxy.LOGGER, "log") as log:
            request, plan = self.proxy.prepare(self.body, "new-private-session")

        self.assertEqual([call[0] for call in calls[:2]], ["/apply-template", "/tokenize"])
        self.assertEqual(calls[1][1], {"content": "rendered-private-prompt", "add_special": False, "parse_special": True})
        self.assertNotIn("id_slot", request)
        self.proxy._restore.assert_not_called()
        self.assertNotEqual(longest, short)
        self.assertFalse(plan.prefix_was_present)
        self.assertEqual(plan.stable_prefix_seed_tokens, (1, 2, 3))
        rendered = str(log.call_args_list)
        self.assertIn("shared_prefix_overlap_discovered", rendered)
        self.assertNotIn("shared_prefix_candidate_restored", rendered)
        self.assertNotIn("shared_prefix_effective_hit", rendered)
        self.assertNotIn("shared_prefix_rejected_by_llama", rendered)
        self.assertNotIn('"cache_hit"', rendered)

    def test_divergent_shared_candidate_never_falls_back_to_unscoped_legacy_prefix(self):
        self._write_shared("local-llm-prefix-divergent.bin", [1, 2, 3])
        legacy = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        legacy.write_bytes(b"legacy")
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 4, 5))
        self.proxy._restore = Mock(return_value=3)

        request, plan = self.proxy.prepare(self.body, "new-session")

        self.assertNotIn("id_slot", request)
        self.proxy._restore.assert_not_called()
        self.assertEqual(plan.stable_prefix_seed_tokens, (1, 2))
        self.assertFalse(plan.prefix_was_present)

    def test_scoped_proxy_never_restores_wrong_scope_snapshot_as_legacy(self):
        scoped = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            shared_prefix_scope="personal",
            minimum_shared_prefix_tokens=2,
        )
        prefix_file = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        prefix_file.write_bytes(b"wrong-scope")
        manifest = PrefixManifest("default", "work", prefix_file.name, (1, 2, 3))
        prefix_file.with_name(manifest_filename(prefix_file.name)).write_text(
            manifest.to_json()
        )
        scoped._render_tokens = Mock(return_value=(1, 2, 3, 4))
        scoped._restore = Mock(return_value=3)

        request, plan = scoped.prepare(self.body, "personal-session")

        self.assertNotIn("id_slot", request)
        scoped._restore.assert_not_called()
        self.assertFalse(plan.prefix_was_present)

    def test_prepare_prefers_complete_shared_prefix_over_tied_divergent_candidate(self):
        self._write_shared("local-llm-prefix-divergent.bin", [1, 2, 3, 4])
        complete = self._write_shared("local-llm-prefix-complete.bin", [1, 2, 3])
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3, 9))
        self.proxy._restore = Mock(return_value=3)

        request, plan = self.proxy.prepare(self.body, "new-session")

        self.assertEqual(request["id_slot"], 0)
        self.proxy._restore.assert_called_once_with(0, complete)
        self.assertTrue(plan.prefix_was_present)
        self.assertIsNone(plan.stable_prefix_seed_tokens)

    def test_shared_restore_prevents_exact_reseed_at_finish(self):
        self._write_shared("local-llm-prefix-shared.bin", [1, 2, 3])
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3, 4))
        self.proxy._restore = Mock(return_value=3)
        self.proxy._save = Mock(return_value=4)
        self.proxy._schedule_prefix_seed = Mock()

        _request, plan = self.proxy.prepare(self.body, "new-session")
        self.proxy.finish(plan, 200)

        self.proxy._schedule_prefix_seed.assert_not_called()

    def test_shared_candidate_logs_effective_outcome_only_after_llama_metadata(self):
        plan = SnapshotPlan(
            "session-a",
            0,
            "prefix",
            Path(self.tempdir.name, "session.bin"),
            Path(self.tempdir.name, "prefix.bin"),
            {},
            True,
            shared_prefix_candidate_tokens=3,
            shared_prefix_verified_lcp=3,
        )
        self.proxy._save = Mock(return_value=4)
        self.proxy._schedule_prefix_seed = Mock()

        with patch.object(cache_proxy.LOGGER, "log") as log:
            self.proxy.finish(plan, 200, cached_tokens=3)
        rendered = str(log.call_args_list)
        self.assertIn("shared_prefix_effective_hit", rendered)
        self.assertNotIn("shared_prefix_rejected_by_llama", rendered)
        self.proxy._schedule_prefix_seed.assert_not_called()

        with patch.object(cache_proxy.LOGGER, "log") as log:
            self.proxy.finish(plan, 200, cached_tokens=0)
        rendered = str(log.call_args_list)
        self.assertIn("shared_prefix_rejected_by_llama", rendered)
        self.assertNotIn("shared_prefix_effective_hit", rendered)
        self.proxy._schedule_prefix_seed.assert_called_once()

    def test_shared_restore_requires_manifest_token_count(self):
        snapshot = self._write_shared("local-llm-prefix-shared.bin", [1, 2, 3])
        manifest = Path(self.tempdir.name, manifest_filename(snapshot.name))
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3, 4))
        self.proxy._restore = Mock(return_value=2)

        request, plan = self.proxy.prepare(self.body, "new-session")

        self.assertNotIn("id_slot", request)
        self.assertFalse(plan.prefix_was_present)
        self.assertFalse(snapshot.exists())
        self.assertFalse(manifest.exists())

    def test_shared_restore_cleanup_error_still_falls_back_cold(self):
        self._write_shared("local-llm-prefix-shared.bin", [1, 2, 3])
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3, 4))
        self.proxy._restore = Mock(return_value=2)

        with patch.object(Path, "unlink", side_effect=PermissionError("read-only")) as unlink:
            request, plan = self.proxy.prepare(self.body, "session-a")

        self.assertNotIn("id_slot", request)
        self.assertIsNone(plan.slot_id)
        self.assertFalse(plan.prefix_was_present)
        self.assertEqual(unlink.call_count, 2)

    def test_rejected_exact_shared_source_is_not_retried_as_legacy(self):
        exact_name = cache_filename("prefix", self.body, "prefix")
        self._write_shared(exact_name, [1, 2, 3])
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3, 4))
        self.proxy._restore = Mock(return_value=2)

        with patch.object(Path, "unlink", side_effect=PermissionError("read-only")):
            request, plan = self.proxy.prepare(self.body, "session-a")

        self.assertNotIn("id_slot", request)
        self.assertIsNone(plan.slot_id)
        self.assertFalse(plan.prefix_was_present)
        self.proxy._restore.assert_called_once()

    def test_session_restore_without_golden_still_schedules_first_seed(self):
        session_file = Path(self.tempdir.name, cache_filename("returning", self.body, "session"))
        session_file.write_bytes(b"session")
        self.proxy._restore = Mock(return_value=3)
        self.proxy._save = Mock(return_value=4)
        self.proxy._schedule_prefix_seed = Mock()

        _request, plan = self.proxy.prepare(self.body, "returning")
        self.proxy.finish(plan, 200)

        self.proxy._schedule_prefix_seed.assert_called_once()

    def test_hot_session_without_golden_still_schedules_first_seed(self):
        key = cache_key(self.body)
        self.proxy.session_states["hot"] = SlotState(0, key, 4)
        self.proxy._slots = Mock(return_value=[{"id": 0, "is_processing": False, "n_prompt_tokens": 4}])
        self.proxy._save = Mock(return_value=4)
        self.proxy._schedule_prefix_seed = Mock()

        _request, plan = self.proxy.prepare(self.body, "hot")
        self.proxy.finish(plan, 200)

        self.proxy._schedule_prefix_seed.assert_called_once()

    def test_shared_prefix_rejects_changed_early_tokens_short_lcp_and_wrong_scope(self):
        self._write_shared("local-llm-prefix-changed.bin", [1, 7, 3])
        self._write_shared("local-llm-prefix-wrong-scope.bin", [1, 2, 3], scope="other")
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 9))
        self.proxy._restore = Mock(return_value=3)

        request, _plan = self.proxy.prepare(self.body, "session-new")

        self.assertNotIn("id_slot", request)
        self.proxy._restore.assert_not_called()

    def test_shared_prefix_ignores_malformed_orphan_mismatched_and_oversized_manifests(self):
        Path(self.tempdir.name, "local-llm-prefix-bad.bin.manifest.json").write_text("not-json")
        orphan_name = "local-llm-prefix-missing.bin"
        orphan = PrefixManifest("default", "default", orphan_name, (1, 2, 3))
        Path(self.tempdir.name, manifest_filename(orphan_name)).write_text(orphan.to_json())
        alias_name = "local-llm-prefix-alias.bin"
        alias = PrefixManifest("default", "default", "local-llm-prefix-other.bin", (1, 2, 3))
        Path(self.tempdir.name, manifest_filename(alias_name)).write_text(alias.to_json())
        Path(self.tempdir.name, "local-llm-prefix-oversized.bin.manifest.json").write_bytes(
            b"x" * (MAX_METADATA_BYTES + 1)
        )
        self.proxy.minimum_shared_prefix_tokens = 1
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3))
        self.proxy._restore = Mock(return_value=3)

        request, _plan = self.proxy.prepare(self.body, "session-new")

        self.assertNotIn("id_slot", request)
        self.proxy._restore.assert_not_called()

    def test_failed_shared_restore_does_not_suppress_first_exact_seed(self):
        self._write_shared("local-llm-prefix-shared.bin", [1, 2, 3])
        self.proxy.minimum_shared_prefix_tokens = 2
        self.proxy._render_tokens = Mock(return_value=(1, 2, 4))
        self.proxy._restore = Mock(side_effect=RuntimeError("incompatible"))

        request, plan = self.proxy.prepare(self.body, "new-session")

        self.assertNotIn("id_slot", request)
        self.assertFalse(plan.prefix_was_present)

    def test_render_tokens_rejects_non_object_tokenizer_response(self):
        self.proxy._json_request = Mock(side_effect=[{"prompt": "rendered"}, [1, 2, 3]])

        with self.assertRaises(TypeError):
            self.proxy._render_tokens(self.body)

    def test_render_or_tokenize_failure_falls_back_to_exact_legacy_prefix_without_leaking_data(self):
        prefix = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        prefix.write_bytes(b"legacy")
        private = "rendered-secret-that-must-not-be-logged"
        self.proxy._json_request = Mock(side_effect=[{"prompt": private}, RuntimeError("tokenizer unavailable")])
        self.proxy._restore = Mock(return_value=2)

        with patch.object(cache_proxy.LOGGER, "log") as log:
            request, plan = self.proxy.prepare(self.body, "private-session")

        self.assertEqual(request["id_slot"], 0)
        self.proxy._restore.assert_called_once_with(0, prefix)
        rendered_logs = str(log.call_args_list)
        self.assertNotIn(private, rendered_logs)
        self.assertNotIn("private-session", rendered_logs)
        self.assertFalse(plan.prefix_was_present)

    def test_legacy_exact_prefix_restore_schedules_golden_migration(self):
        prefix = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        prefix.write_bytes(b"legacy")
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3))
        self.proxy._restore = Mock(return_value=3)
        self.proxy._save = Mock(return_value=4)
        self.proxy._schedule_prefix_seed = Mock()

        _request, plan = self.proxy.prepare(self.body, "new-session")
        self.proxy.finish(plan, 200)

        self.proxy._schedule_prefix_seed.assert_called_once()

    def test_session_snapshot_is_saved_and_restored(self):
        request, plan = self.proxy.prepare(self.body, "session-a")
        self.assertNotIn("id_slot", request)
        self.proxy.finish(plan, 200)

        prefix_file = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        prefix_file.write_bytes(b"prefix snapshot")
        files = {path.name for path in Path(self.tempdir.name).glob("local-llm-*.bin")}
        self.assertEqual(len(files), 2)

        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        cold_proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=1,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        request, _ = cold_proxy.prepare(other_body, "session-b")
        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(len(FakeLlamaHandler.restored), 1)

    def test_incompatible_snapshot_falls_back_to_current_slot(self):
        request, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        prefix_file = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        prefix_file.write_bytes(b"prefix snapshot")
        FakeLlamaHandler.restore_failed = True

        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        cold_proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=1,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        _request, plan = cold_proxy.prepare(other_body, "session-b")

        self.assertEqual(len(FakeLlamaHandler.restored), 1)

    def test_same_session_reuses_hot_slot_without_disk_restore(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        restored_before = len(FakeLlamaHandler.restored)

        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        request, _ = self.proxy.prepare(other_body, "session-a")

        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(len(FakeLlamaHandler.restored), restored_before)

    def test_invalid_upstream_url_is_rejected(self):
        with self.assertRaises(ValueError):
            LlamaCacheProxy(upstream="https://example.test")

    def test_invalid_shared_prefix_configuration_is_rejected(self):
        kwargs: dict[str, Any] = {
            "upstream": f"http://127.0.0.1:{self.server.server_port}",
            "cache_dir": self.tempdir.name,
        }
        with self.assertRaises(ValueError):
            LlamaCacheProxy(**kwargs, shared_prefix_scope=" ")
        for minimum in (True, 0, 1.5):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                LlamaCacheProxy(**kwargs, minimum_shared_prefix_tokens=minimum)
        for timeout in (True, False, 0, -1, float("nan"), float("inf"), float("-inf"), "600", None):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                LlamaCacheProxy(
                    **kwargs,
                    prefix_seed_timeout_seconds=cast(Any, timeout),
                )

    def test_new_seed_timeout_preserves_old_positional_constructor_contract(self):
        proxy = LlamaCacheProxy(
            f"http://127.0.0.1:{self.server.server_port}",
            self.tempdir.name,
            1,
            120,
            True,
            2,
            "terminal",
        )

        self.assertEqual(proxy.save_policy, "terminal")
        self.assertEqual(proxy.prefix_seed_timeout_seconds, 600.0)

    def test_finish_ignores_unsuccessful_response(self):
        _, plan = self.proxy.prepare(self.body, "session-a")

        self.proxy.finish(plan, 500)

        self.assertEqual(self.proxy.session_states, {})
        self.assertEqual(FakeLlamaHandler.saved, [])

    def test_terminal_save_policy_defers_tool_calls_but_flushes_before_eviction(self):
        self.proxy.save_policy = "terminal"
        _, plan = self.proxy.prepare(self.body, "private-session-a")
        FakeLlamaHandler.slot_tokens = 12

        self.proxy.finish(plan, 200, finish_reason="tool_calls")

        state = self.proxy.session_states["private-session-a"]
        self.assertTrue(state.dirty)
        self.assertEqual(state.n_tokens, 12)
        self.assertEqual(FakeLlamaHandler.saved, [])

        next_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "session b"}],
        }
        self.proxy.prepare(next_body, "private-session-b")

        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        self.assertFalse(self.proxy.session_states["private-session-a"].dirty)

    def test_terminal_save_policy_keeps_dirty_session_hot_and_saves_terminal_response(self):
        self.proxy.save_policy = "terminal"
        _, plan = self.proxy.prepare(self.body, "session-a")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(plan, 200, finish_reason="tool_calls")

        request, hot_plan = self.proxy.prepare(self.body, "session-a")
        self.assertEqual(request["id_slot"], 0)
        self.assertTrue(self.proxy.session_states["session-a"].dirty)

        FakeLlamaHandler.slot_tokens = 15
        self.proxy.finish(hot_plan, 200, finish_reason="stop")

        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        self.assertFalse(self.proxy.session_states["session-a"].dirty)

    def test_compacted_prompt_with_same_affinity_reuses_only_a_compatible_hot_prefix(self):
        self.proxy.save_policy = "terminal"
        _, plan = self.proxy.prepare(self.body, "session-parent")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(plan, 200, finish_reason="tool_calls")
        compacted = {
            **self.body,
            "messages": [
                self.body["messages"][0],
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "continue"},
            ],
        }

        request, compacted_plan = self.proxy.prepare(compacted, "session-parent")

        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(compacted_plan.prefix_key, plan.prefix_key)
        self.assertTrue(self.proxy.session_states["session-parent"].dirty)
        self.assertEqual(FakeLlamaHandler.saved, [])

    def test_compaction_session_rotation_never_restores_parent_snapshot(self):
        _, parent_plan = self.proxy.prepare(self.body, "session-parent")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(parent_plan, 200, finish_reason="stop")
        parent_snapshot = parent_plan.session_file
        compacted = {
            **self.body,
            "messages": [
                self.body["messages"][0],
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "continue"},
            ],
        }
        restored_before = list(FakeLlamaHandler.restored)

        request, child_plan = self.proxy.prepare(compacted, "session-child")

        self.assertNotIn("id_slot", request)
        self.assertNotEqual(child_plan.session_file, parent_snapshot)
        self.assertEqual(FakeLlamaHandler.restored, restored_before)

    def test_in_place_compaction_after_eviction_restores_same_session_for_llama_lcp(self):
        _, parent_plan = self.proxy.prepare(self.body, "session-parent")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(parent_plan, 200, finish_reason="stop")
        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "other"}],
        }
        _, other_plan = self.proxy.prepare(other_body, "session-other")
        self.proxy.finish(other_plan, 200, finish_reason="stop")
        compacted = {
            **self.body,
            "messages": [
                self.body["messages"][0],
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "continue"},
            ],
        }

        request, compacted_plan = self.proxy.prepare(compacted, "session-parent")

        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(compacted_plan.session_file, parent_plan.session_file)
        self.assertEqual(FakeLlamaHandler.restored[-1], parent_plan.session_file.name)

    def test_compaction_that_changes_stable_prefix_flushes_then_starts_cold(self):
        self.proxy.save_policy = "terminal"
        _, plan = self.proxy.prepare(self.body, "session-parent")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(plan, 200, finish_reason="tool_calls")
        compacted = {
            **self.body,
            "messages": [
                {"role": "system", "content": "new deployment rules"},
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            ],
        }

        request, compacted_plan = self.proxy.prepare(compacted, "session-parent")

        self.assertNotIn("id_slot", request)
        self.assertNotEqual(compacted_plan.prefix_key, plan.prefix_key)
        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        self.assertFalse(self.proxy.session_states["session-parent"].dirty)
        self.assertEqual(FakeLlamaHandler.restored, [])

    def test_all_save_policy_still_saves_tool_call_responses(self):
        self.proxy.save_policy = "all"
        _, plan = self.proxy.prepare(self.body, "session-a")

        self.proxy.finish(plan, 200, finish_reason="tool_calls")

        self.assertEqual(len(FakeLlamaHandler.saved), 1)

    def test_terminal_policy_saves_when_slot_metadata_is_unavailable(self):
        self.proxy.save_policy = "terminal"
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy._slot_token_count = Mock(return_value=0)

        self.proxy.finish(plan, 200, finish_reason="tool_calls")

        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        self.assertFalse(self.proxy.session_states["session-a"].dirty)

    def test_terminal_policy_saves_when_slot_metadata_request_fails(self):
        self.proxy.save_policy = "terminal"
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy._slots = Mock(side_effect=RuntimeError("slots unavailable"))

        self.proxy.finish(plan, 200, finish_reason="tool_calls")

        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        self.assertFalse(self.proxy.session_states["session-a"].dirty)

    def test_flush_dirty_ignores_clean_or_unpersistable_states(self):
        self.proxy.session_states = {
            "clean": SlotState(0, "clean", 10, Path(self.tempdir.name, "clean.bin"), False),
            "missing-file": SlotState(1, "dirty", 10, None, True),
        }
        self.proxy._save = Mock()

        self.proxy._flush_dirty_states("test")

        self.proxy._save.assert_not_called()

    def test_flush_dirty_public_method_persists_dirty_state(self):
        target = Path(self.tempdir.name, "session.bin")
        self.proxy.session_states = {
            "session-a": SlotState(0, "prefix", 12, target, True),
        }

        self.proxy.flush_dirty("shutdown")

        self.assertFalse(self.proxy.session_states["session-a"].dirty)
        self.assertTrue(target.exists())

    def test_slot_token_count_handles_mismatch_and_missing_slot(self):
        self.proxy._slots = Mock(
            return_value=[
                {"id": 1, "is_processing": False, "n_prompt_tokens": 7},
                {"id": 0, "is_processing": True, "n_prompt_tokens": 8},
            ]
        )
        self.assertEqual(self.proxy._slot_token_count(0), 0)
        self.assertEqual(self.proxy._slot_token_count(2), 0)

    def test_slot_resolution_does_not_guess_without_a_candidate(self):
        plan = SnapshotPlan(
            "session-a",
            None,
            "prefix",
            Path(self.tempdir.name, "session.bin"),
            Path(self.tempdir.name, "prefix.bin"),
            {},
            False,
        )
        self.proxy._slots = Mock(side_effect=RuntimeError("slots unavailable"))

        with self.assertRaises(RuntimeError):
            self.proxy._resolve_slot(plan)

    def test_invalid_save_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            LlamaCacheProxy(
                upstream=f"http://127.0.0.1:{self.server.server_port}",
                cache_dir=self.tempdir.name,
                save_policy="unsafe",
            )

    def test_cache_directory_is_private(self):
        cache_dir = Path(self.tempdir.name, "private-cache")

        LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=str(cache_dir),
            enable_prefix_seeding=False,
        )

        self.assertEqual(cache_dir.stat().st_mode & 0o777, 0o700)

    def test_finish_does_not_seed_when_prefix_snapshot_is_present(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        plan = SnapshotPlan(
            plan.session_id,
            plan.slot_id,
            plan.prefix_key,
            plan.session_file,
            plan.prefix_file,
            plan.prefix_payload,
            True,
        )
        self.proxy._schedule_prefix_seed = Mock()

        self.proxy.finish(plan, 200)

        self.proxy._schedule_prefix_seed.assert_not_called()

    def test_hot_state_rejects_stale_slot_states(self):
        state = SlotState(1, cache_key(self.body), 10)

        self.assertIsNone(self.proxy._hot_state(None, state.prefix_key))
        self.assertIsNone(self.proxy._hot_state(state, "different-prefix"))

        with patch.object(self.proxy, "_slots", return_value=[{"id": 0, "is_processing": False}]):
            self.assertIsNone(self.proxy._hot_state(state, state.prefix_key))
        with patch.object(self.proxy, "_slots", return_value=[{"id": 1, "is_processing": True}]):
            self.assertIsNone(self.proxy._hot_state(state, state.prefix_key))
        with patch.object(
            self.proxy,
            "_slots",
            return_value=[{"id": 1, "is_processing": False, "n_prompt_tokens": 9}],
        ):
            self.assertIsNone(self.proxy._hot_state(state, state.prefix_key))
        with patch.object(
            self.proxy,
            "_slots",
            return_value=[{"id": 1, "is_processing": False, "n_prompt_tokens": 10}],
        ):
            self.assertEqual(self.proxy._hot_state(state, state.prefix_key), state)

    def test_forget_slot_removes_only_that_session(self):
        self.proxy.session_states = {
            "session-a": SlotState(0, "a", 1),
            "session-b": SlotState(1, "b", 2),
        }

        self.proxy._forget_slot(0)

        self.assertEqual(list(self.proxy.session_states), ["session-b"])

    def test_schedule_prefix_seed_skips_empty_and_duplicate_work(self):
        empty_plan = SnapshotPlan(
            "session-a",
            0,
            "key",
            Path(self.tempdir.name, "session.bin"),
            Path(self.tempdir.name, "prefix.bin"),
            {"messages": [], "tools": []},
            False,
        )
        self.proxy.enable_prefix_seeding = True
        with patch("cache_proxy.threading.Thread") as thread:
            self.proxy._schedule_prefix_seed(empty_plan)
            thread.assert_not_called()

        tools_only_plan = SnapshotPlan(
            "session-a",
            0,
            "key",
            empty_plan.session_file,
            empty_plan.prefix_file,
            {"messages": [], "tools": [{"type": "function"}]},
            False,
        )
        with patch("cache_proxy.threading.Thread") as thread:
            self.proxy._schedule_prefix_seed(tools_only_plan)
            thread.assert_not_called()

        seeded_plan = SnapshotPlan(
            "session-a",
            0,
            "key",
            empty_plan.session_file,
            empty_plan.prefix_file,
            {"messages": [{"role": "system", "content": "rules"}], "tools": []},
            False,
            stable_prefix_seed_tokens=(1, 2, 3),
        )
        self.proxy.prefix_seeds_in_flight.add(seeded_plan.prefix_file)
        with patch("cache_proxy.threading.Thread") as thread:
            self.proxy._schedule_prefix_seed(seeded_plan)
            thread.assert_not_called()
        self.proxy.prefix_seeds_in_flight.clear()

        with patch("cache_proxy.threading.Thread") as thread:
            thread.return_value.start = Mock()
            self.proxy._schedule_prefix_seed(seeded_plan)
            thread.assert_called_once_with(
                target=self.proxy._seed_prefix,
                args=(seeded_plan.prefix_payload, seeded_plan.prefix_file, 0, (1, 2, 3)),
                name="local-llm-kv-prefix-seed",
                daemon=True,
            )
            thread.return_value.start.assert_called_once_with()

    def test_deferred_stable_seed_remains_pending_and_retries_after_dirty_slot(self):
        self.proxy.enable_prefix_seeding = True
        self.proxy.prefix_seed_delay_seconds = 0
        plan = SnapshotPlan(
            "session-a",
            0,
            "key",
            Path(self.tempdir.name, "session.bin"),
            Path(self.tempdir.name, "prefix.bin"),
            {"messages": [{"role": "system", "content": "rules"}], "tools": []},
            False,
            stable_prefix_seed_tokens=(1, 2, 3),
        )

        with patch("cache_proxy.threading.Thread") as thread:
            thread.return_value.start = Mock()
            self.proxy._schedule_prefix_seed(plan)
            args = thread.call_args.kwargs["args"]
        self.proxy.prefix_seeds_in_flight.clear()
        self.proxy.session_states["session-a"] = SlotState(
            0,
            "key",
            10,
            plan.session_file,
            True,
        )
        self.proxy._slots = Mock(
            return_value=[{"id": 0, "is_processing": False, "n_prompt_tokens": 10}]
        )

        self.proxy._seed_prefix(*args)

        self.assertIn(plan.prefix_file, self.proxy.pending_prefix_seeds)
        self.assertNotIn(plan.prefix_file, self.proxy.prefix_seeds_in_flight)
        later_plan = replace(plan, stable_prefix_seed_tokens=None)
        with patch.object(self.proxy, "_start_pending_prefix_seed"):
            self.proxy._schedule_prefix_seed(later_plan)
        self.assertEqual(
            self.proxy.pending_prefix_seeds[plan.prefix_file].stable_tokens,
            (1, 2, 3),
        )
        with patch("cache_proxy.threading.Thread") as retry_thread:
            retry_thread.return_value.start = Mock()
            self.proxy._retry_pending_prefix_seeds()
            retry_thread.assert_called_once()

    def test_pending_seed_queue_applies_backpressure_without_losing_accepted_work(self):
        self.proxy.enable_prefix_seeding = True
        plans = [
            SnapshotPlan(
                f"session-{index}",
                0,
                "key",
                Path(self.tempdir.name, f"session-{index}.bin"),
                Path(self.tempdir.name, f"prefix-{index}.bin"),
                {"messages": [{"role": "system", "content": "rules"}], "tools": []},
                False,
            )
            for index in range(17)
        ]

        with (
            patch.object(self.proxy, "_start_pending_prefix_seed") as start_seed,
            patch("cache_proxy._log_event") as log_event,
        ):
            for plan in plans[:16]:
                self.proxy._schedule_prefix_seed(plan)
            accepted = dict(self.proxy.pending_prefix_seeds)

            self.proxy._schedule_prefix_seed(plans[16])

            self.assertEqual(self.proxy.pending_prefix_seeds, accepted)
            self.assertNotIn(plans[16].prefix_file, self.proxy.pending_prefix_seeds)
            self.assertEqual(start_seed.call_count, 16)
            log_event.assert_called_once_with(
                "prefix_seed_queue_full",
                level=cache_proxy.logging.WARNING,
                filename=plans[16].prefix_file.name,
                limit=16,
            )

            updated = replace(plans[0], stable_prefix_seed_tokens=(1, 2, 3))
            self.proxy._schedule_prefix_seed(updated)

            self.assertEqual(set(self.proxy.pending_prefix_seeds), set(accepted))
            self.assertEqual(
                self.proxy.pending_prefix_seeds[updated.prefix_file].stable_tokens,
                (1, 2, 3),
            )
            start_seed.assert_called_with(updated.prefix_file)

    def test_completed_old_seed_does_not_discard_newer_longer_pending_seed(self):
        prefix_file = Path(self.tempdir.name, "prefix.bin")
        self.proxy.pending_prefix_seeds[prefix_file] = cache_proxy.PendingPrefixSeed(
            {"messages": [{"role": "system", "content": "rules"}]},
            prefix_file,
            0,
            (1, 2, 3),
        )

        self.proxy._discard_pending_prefix_seed(prefix_file, (1, 2))
        self.assertIn(prefix_file, self.proxy.pending_prefix_seeds)

        self.proxy._discard_pending_prefix_seed(prefix_file, (1, 2, 3))
        self.assertNotIn(prefix_file, self.proxy.pending_prefix_seeds)

    def test_foreground_operation_releases_lock_on_success_and_error(self):
        with self.proxy.foreground_operation():
            self.assertEqual(self.proxy.foreground_waiters, 0)
        self.assertEqual(self.proxy.foreground_waiters, 0)

        class RaisingLock:
            def acquire(self):
                raise RuntimeError("lock failed")

        self.proxy.operation_lock = RaisingLock()
        with self.assertRaises(RuntimeError):
            with self.proxy.foreground_operation():
                pass
        self.assertEqual(self.proxy.foreground_waiters, 0)

        self.proxy.operation_lock = threading.Lock()
        with self.assertRaises(ValueError):
            with self.proxy.foreground_operation():
                raise ValueError("request failed")
        self.assertEqual(self.proxy.foreground_waiters, 0)

    def _run_reserved_request(self, session_id, entered, release, results):
        with self.proxy.session_operation(session_id):
            request, plan = self.proxy.begin_request(self.body, session_id)
            results.append((session_id, request["id_slot"], plan.session_file))
            entered.set()
            self.assertTrue(release.wait(2))
            self.proxy.complete_request(plan, 200)

    def test_distinct_sessions_forward_concurrently_on_separate_slots_without_snapshot_crossing(self):
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 0},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 0},
        ]
        self.proxy._slots = Mock(return_value=slots)
        self.proxy._render_tokens = Mock(side_effect=ValueError("no shared prefix"))
        saves = []

        def save(slot_id, target, **_kwargs):
            saves.append((slot_id, target))
            return 10

        self.proxy._save = Mock(side_effect=save)
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()
        results = []
        threads = [
            threading.Thread(
                target=self._run_reserved_request,
                args=(session_id, entered[index], release, results),
            )
            for index, session_id in enumerate(("parallel-a", "parallel-b"))
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(entered[0].wait(2))
        self.assertTrue(entered[1].wait(2))
        self.assertEqual({slot_id for _session, slot_id, _path in results}, {0, 1})
        release.set()
        for thread in threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())

        expected = {session: (slot_id, path) for session, slot_id, path in results}
        self.assertEqual(
            {(slot_id, target) for slot_id, target in saves},
            {(slot_id, path) for slot_id, path in expected.values()},
        )
        self.assertEqual(
            {session: state.slot_id for session, state in self.proxy.session_states.items()},
            {session: slot_id for session, (slot_id, _path) in expected.items()},
        )

    def test_same_session_requests_remain_serialized(self):
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 0},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 0},
        ]
        self.proxy._slots = Mock(return_value=slots)
        self.proxy._render_tokens = Mock(side_effect=ValueError("no shared prefix"))
        self.proxy._save = Mock(return_value=10)
        first_entered = threading.Event()
        second_entered = threading.Event()
        first_release = threading.Event()
        second_release = threading.Event()
        results = []
        first = threading.Thread(
            target=self._run_reserved_request,
            args=("same-session", first_entered, first_release, results),
        )
        second = threading.Thread(
            target=self._run_reserved_request,
            args=("same-session", second_entered, second_release, results),
        )
        first.start()
        self.assertTrue(first_entered.wait(2))
        second.start()
        self.assertFalse(second_entered.wait(0.2))
        first_release.set()
        first.join(2)
        self.assertTrue(second_entered.wait(2))
        second_release.set()
        second.join(2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())

    def test_preparing_parallel_session_never_flushes_active_slot_snapshot(self):
        active_snapshot = Path(self.tempdir.name, "active-session.bin")
        self.proxy.session_states["active"] = SlotState(
            0,
            cache_key(self.body),
            12,
            active_snapshot,
            True,
        )
        self.proxy.active_slot_ids.add(0)
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": True, "n_prompt_tokens": 12},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 0},
            ]
        )
        self.proxy._render_tokens = Mock(side_effect=ValueError("no shared prefix"))
        self.proxy._save = Mock()

        request, plan = self.proxy.prepare(
            self.body,
            "parallel",
            excluded_slot_ids={0},
            wait_for_slot=False,
        )

        self.assertNotIn("id_slot", request)
        self.assertEqual(plan.candidate_slot_id, 1)
        self.proxy._save.assert_not_called()

    def test_single_slot_serializes_distinct_sessions(self):
        self.proxy._slots = Mock(
            return_value=[{"id": 0, "is_processing": False, "n_prompt_tokens": 0}]
        )
        self.proxy._render_tokens = Mock(side_effect=ValueError("no shared prefix"))
        self.proxy._save = Mock(return_value=10)
        first_entered = threading.Event()
        second_entered = threading.Event()
        first_release = threading.Event()
        second_release = threading.Event()
        results = []
        first = threading.Thread(
            target=self._run_reserved_request,
            args=("single-a", first_entered, first_release, results),
        )
        second = threading.Thread(
            target=self._run_reserved_request,
            args=("single-b", second_entered, second_release, results),
        )
        first.start()
        self.assertTrue(first_entered.wait(2))
        second.start()
        self.assertFalse(second_entered.wait(0.2))
        first_release.set()
        first.join(2)
        self.assertTrue(second_entered.wait(2))
        second_release.set()
        second.join(2)
        self.assertEqual([slot_id for _session, slot_id, _path in results], [0, 0])

    def test_prefix_seed_cannot_race_reserved_foreground_slot(self):
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": False, "n_prompt_tokens": 0},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 0},
            ]
        )
        self.proxy._render_tokens = Mock(side_effect=ValueError("no shared prefix"))
        with self.proxy.session_operation("foreground"):
            request, plan = self.proxy.begin_request(self.body, "foreground")
            self.assertIn("id_slot", request)
            self.proxy.prefix_seed_delay_seconds = 0
            self.proxy._json_request = Mock()
            self.proxy._seed_prefix(self.body, Path(self.tempdir.name, "prefix.bin"), plan.slot_id)
            self.proxy._json_request.assert_not_called()
            self.proxy.cancel_request(plan)

    def test_long_seed_on_spare_slot_does_not_block_hot_owner_foreground(self):
        self.proxy.prefix_seed_delay_seconds = 0
        owner_snapshot = Path(self.tempdir.name, "owner.bin")
        owner_snapshot.write_bytes(b"known-hot-owner")
        owner_state = SlotState(0, cache_key(self.body), 10, owner_snapshot, False)
        self.proxy.session_states["owner"] = owner_state
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": False, "n_prompt_tokens": 10},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 0},
            ]
        )
        seed_entered_completion = threading.Event()
        release_seed = threading.Event()
        foreground_acquired = threading.Event()
        foreground_result = []

        def block_seed(_method, path, _payload, **_kwargs):
            self.assertEqual(path, "/completion")
            seed_entered_completion.set()
            self.assertTrue(release_seed.wait(2))
            return {}

        def save_seed(slot_id, target, **_kwargs):
            self.assertEqual(slot_id, 1)
            target.write_bytes(b"seed")
            return 3

        def run_foreground():
            with self.proxy.session_operation("owner"):
                request, plan = self.proxy.begin_request(self.body, "owner")
                foreground_result.append((request["id_slot"], plan))
                foreground_acquired.set()
                self.proxy.cancel_request(plan)

        self.proxy._json_request = Mock(side_effect=block_seed)
        self.proxy._save = Mock(side_effect=save_seed)
        seed_thread = threading.Thread(
            target=self.proxy._seed_prefix,
            args=(self.body, Path(self.tempdir.name, "prefix.bin"), 0, (1, 2, 3)),
        )
        foreground_thread = threading.Thread(target=run_foreground)
        seed_thread.start()
        try:
            self.assertTrue(seed_entered_completion.wait(1))
            self.assertEqual(self.proxy.seed_slot_ids, {1})
            foreground_thread.start()
            self.assertTrue(
                foreground_acquired.wait(0.5),
                "foreground stayed blocked while the spare-slot seed RPC was running",
            )
        finally:
            release_seed.set()
            seed_thread.join(2)
            foreground_thread.join(2)

        self.assertFalse(seed_thread.is_alive())
        self.assertFalse(foreground_thread.is_alive())
        self.assertEqual(foreground_result[0][0], 0)
        self.assertEqual(self.proxy.session_states["owner"], owner_state)
        self.assertEqual(owner_snapshot.read_bytes(), b"known-hot-owner")
        self.assertEqual(self.proxy.seed_slot_ids, set())

    def test_failed_seed_releases_slot_and_token_reservations_but_keeps_pending(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": False, "n_prompt_tokens": 0},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 0},
            ]
        )
        prefix_file = Path(self.tempdir.name, "prefix-failure.bin")
        pending = cache_proxy.PendingPrefixSeed(self.body, prefix_file, 0, (1, 2, 3))
        self.proxy.pending_prefix_seeds[prefix_file] = pending
        self.proxy.prefix_seeds_in_flight.add(prefix_file)
        self.proxy._json_request = Mock(side_effect=RuntimeError("seed failed"))
        self.proxy._save = Mock()
        self.proxy._prune = Mock()

        self.proxy._seed_prefix(self.body, prefix_file, 0, (1, 2, 3))

        self.assertEqual(self.proxy.seed_slot_ids, set())
        self.assertEqual(self.proxy.seed_slot_snapshots, {})
        self.assertEqual(self.proxy.seed_token_reservations, set())
        self.assertEqual(self.proxy.pending_prefix_seeds[prefix_file], pending)
        self.assertNotIn(prefix_file, self.proxy.prefix_seeds_in_flight)
        self.proxy._save.assert_not_called()

    def test_concurrent_seeds_atomically_reserve_distinct_spare_slots(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy.session_states["owner"] = SlotState(0, cache_key(self.body), 10)
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": False, "n_prompt_tokens": 10},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 0},
                {"id": 2, "is_processing": False, "n_prompt_tokens": 0},
            ]
        )
        entered = {(1, 2, 3): threading.Event(), (4, 5, 6): threading.Event()}
        release = threading.Event()
        completion_slots = []

        def block_seed(_method, path, payload, **_kwargs):
            self.assertEqual(path, "/completion")
            tokens = tuple(payload["prompt"])
            completion_slots.append(payload["id_slot"])
            entered[tokens].set()
            self.assertTrue(release.wait(2))
            return {}

        def save_seed(_slot_id, target, **_kwargs):
            target.write_bytes(b"seed")
            return 3

        self.proxy._json_request = Mock(side_effect=block_seed)
        self.proxy._save = Mock(side_effect=save_seed)
        threads = [
            threading.Thread(
                target=self.proxy._seed_prefix,
                args=(
                    self.body,
                    Path(self.tempdir.name, f"prefix-{index}.bin"),
                    0,
                    tokens,
                ),
            )
            for index, tokens in enumerate(((1, 2, 3), (4, 5, 6)))
        ]
        threads[0].start()
        try:
            self.assertTrue(entered[(1, 2, 3)].wait(1))
            threads[1].start()
            self.assertTrue(entered[(4, 5, 6)].wait(1))
            self.assertEqual(self.proxy.seed_slot_ids, {1, 2})
            self.assertEqual(set(completion_slots), {1, 2})
        finally:
            release.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(self.proxy.seed_slot_ids, set())
        self.assertEqual(self.proxy.seed_token_reservations, set())

    def test_session_snapshot_wins_after_switching_same_prefix_session(self):
        _, first_plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(first_plan, 200)
        _, second_plan = self.proxy.prepare(self.body, "session-b")
        self.proxy.finish(second_plan, 200)
        FakeLlamaHandler.restored = []

        request, _ = self.proxy.prepare(self.body, "session-a")

        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(
            FakeLlamaHandler.restored,
            [cache_filename("session-a", self.body, "session")],
        )

    def test_new_session_uses_unowned_idle_slot_before_evicting_session(self):
        self.proxy.session_states["session-a"] = SlotState(1, cache_key(self.body), 10)
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 0},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 10},
        ]

        with patch.object(self.proxy, "_slots", return_value=slots):
            request, _ = self.proxy.prepare(self.body, "session-b")

        self.assertNotIn("id_slot", request)

    def test_new_session_leaves_slot_selection_to_llama(self):
        request, plan = self.proxy.prepare(self.body, "session-new")

        self.assertNotIn("id_slot", request)
        self.assertIsNone(plan.slot_id)
        self.assertIsNotNone(plan.candidate_slot_id)

    def test_new_session_pins_the_slot_where_its_snapshot_was_restored(self):
        snapshot = Path(self.tempdir.name, cache_filename("session-new", self.body, "session"))
        snapshot.write_bytes(b"snapshot")
        self.proxy._restore_first_available = Mock(return_value=snapshot)

        request, plan = self.proxy.prepare(self.body, "session-new")

        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(plan.slot_id, 0)
        self.proxy._restore_first_available.assert_called_once_with(
            0,
            (snapshot,),
            session_ref=cache_proxy._session_ref("session-new"),
        )

    def test_fallback_resolves_slot_changed_by_native_scheduler(self):
        plan = SnapshotPlan(
            "session-new",
            None,
            "prefix",
            Path(self.tempdir.name, "session.bin"),
            Path(self.tempdir.name, "prefix.bin"),
            {"messages": []},
            True,
            0,
            {0: 10, 1: 20},
        )
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "id_task": 11, "is_processing": False, "n_prompt_tokens": 10},
                {"id": 1, "id_task": 20, "is_processing": False, "n_prompt_tokens": 20},
            ]
        )

        self.assertEqual(self.proxy._resolve_slot(plan), 0)

    def test_fallback_slot_resolution_handles_pinned_and_waiting_states(self):
        pinned = SnapshotPlan(
            "session-a",
            1,
            "prefix",
            Path(self.tempdir.name, "session.bin"),
            Path(self.tempdir.name, "prefix.bin"),
            {"messages": []},
            True,
        )
        self.assertEqual(self.proxy._resolve_slot(pinned), 1)

        waiting = SnapshotPlan(
            "session-b",
            None,
            "prefix",
            pinned.session_file,
            pinned.prefix_file,
            pinned.prefix_payload,
            True,
            0,
            {0: 1},
        )
        self.proxy._slots = Mock(
            return_value=[{"id": 0, "id_task": 1, "is_processing": True}]
        )
        self.assertEqual(self.proxy._resolve_slot(waiting), 0)

        no_candidate = SnapshotPlan(
            "session-c",
            None,
            "prefix",
            pinned.session_file,
            pinned.prefix_file,
            pinned.prefix_payload,
            True,
        )
        with patch.object(self.proxy, "_wait_for_idle_slot", return_value=1):
            self.assertEqual(self.proxy._resolve_slot(no_candidate), 1)

    def test_restore_tries_next_existing_snapshot_source(self):
        missing = Path(self.tempdir.name, "missing.bin")
        existing = Path(self.tempdir.name, "existing.bin")
        existing.write_bytes(b"snapshot")

        restored = self.proxy._restore_first_available(0, (missing, existing))

        self.assertEqual(restored, existing)
        self.assertEqual(FakeLlamaHandler.restored, ["existing.bin"])

    def test_restore_tries_next_source_after_transport_failure(self):
        first = Path(self.tempdir.name, "first.bin")
        second = Path(self.tempdir.name, "second.bin")
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        self.proxy._restore = Mock(side_effect=[TimeoutError("timeout"), 10])

        restored = self.proxy._restore_first_available(0, (first, second))

        self.assertEqual(restored, second)
        self.assertEqual(self.proxy._restore.call_count, 2)

    def test_zero_token_restore_is_treated_as_failure(self):
        source = Path(self.tempdir.name, "empty.bin")
        source.write_bytes(b"snapshot")
        FakeLlamaHandler.restore_tokens = 0

        restored = self.proxy._restore_first_available(0, (source,))

        self.assertIsNone(restored)

    def test_generic_restore_count_mismatch_keeps_source_when_not_shared(self):
        source = Path(self.tempdir.name, "private-session.bin")
        source.write_bytes(b"snapshot")
        self.proxy._restore = Mock(return_value=2)

        restored = self.proxy._restore_first_available(
            0,
            (source,),
            expected_tokens=3,
        )

        self.assertIsNone(restored)
        self.assertTrue(source.exists())

    def test_hot_session_finish_keeps_pinned_slot(self):
        _, first_plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(first_plan, 200)
        _, hot_plan = self.proxy.prepare(self.body, "session-a")

        self.proxy.finish(hot_plan, 200)

        self.assertEqual(hot_plan.slot_id, 0)
        self.assertEqual(FakeLlamaHandler.saved[-1].endswith(".tmp"), True)

    def test_prefix_seed_skips_when_no_safe_idle_slot_exists(self):
        self.proxy.session_states["session-a"] = SlotState(1, cache_key(self.body), 10)
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": True, "n_prompt_tokens": 10},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 10},
            ]
        )
        self.proxy._json_request = Mock()
        self.proxy._save = Mock()

        self.proxy._seed_prefix(
            {"messages": [{"role": "system", "content": "rules"}]},
            Path(self.tempdir.name, "prefix.bin"),
            excluded_slot_id=1,
            stable_tokens=(1, 2, 3),
        )

        self.proxy._json_request.assert_not_called()
        self.proxy._save.assert_not_called()

    def test_prefix_seed_uses_unowned_idle_slot(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": False, "n_prompt_tokens": 3},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 10},
            ]
        )
        self.proxy._json_request = Mock(return_value={})
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3))
        self.proxy._save = Mock(
            side_effect=lambda _slot, target, **_kwargs: (target.write_bytes(b"snapshot"), 3)[1]
        )
        self.proxy._forget_slot = Mock()
        self.proxy._prune = Mock()
        prefix_file = Path(self.tempdir.name, "local-llm-prefix-test.bin")

        self.proxy._seed_prefix(
            {"messages": [{"role": "system", "content": "rules"}]},
            prefix_file,
            excluded_slot_id=1,
        )

        method, path, request = self.proxy._json_request.call_args.args
        self.assertEqual((method, path), ("POST", "/completion"))
        self.assertEqual(
            request,
            {
                "prompt": [1, 2, 3],
                "cache_prompt": False,
                "n_predict": 0,
                "stream": False,
                "id_slot": 0,
            },
        )
        self.proxy._save.assert_called_once_with(0, prefix_file)
        self.proxy._prune.assert_called_once_with()

    def test_prefix_seed_uses_verified_stable_tokens_without_rerendering_private_payload(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(return_value=[{"id": 0, "is_processing": False}])
        self.proxy._render_tokens = Mock(side_effect=AssertionError("must not rerender"))
        self.proxy._json_request = Mock(return_value={})
        self.proxy._save = Mock(
            side_effect=lambda _slot, target, **_kwargs: (target.write_bytes(b"snapshot"), 3)[1]
        )
        self.proxy._forget_slot = Mock()
        self.proxy._prune = Mock()
        prefix_file = Path(self.tempdir.name, "local-llm-prefix-stable.bin")

        self.proxy._seed_prefix(
            {"messages": [{"role": "system", "content": "private"}]},
            prefix_file,
            excluded_slot_id=1,
            stable_tokens=(1, 2, 3),
        )

        self.proxy._render_tokens.assert_not_called()
        self.proxy._json_request.assert_called_once_with(
            "POST",
            "/completion",
            {
                "prompt": [1, 2, 3],
                "cache_prompt": False,
                "n_predict": 0,
                "stream": False,
                "id_slot": 0,
            },
            timeout=600.0,
        )

    def test_prefix_seed_count_mismatch_removes_unpublished_pair_without_private_logs(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(return_value=[{"id": 0, "is_processing": False}])
        private_prompt = "private prompt must never be logged"
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3))
        self.proxy._json_request = Mock(return_value={})
        prefix_file = Path(self.tempdir.name, "local-llm-prefix-mismatch.bin")
        manifest_file = Path(self.tempdir.name, manifest_filename(prefix_file.name))

        def mismatched_save(_slot, target, **_kwargs):
            target.write_bytes(private_prompt.encode())
            return 2

        self.proxy._save = Mock(side_effect=mismatched_save)
        with patch.object(cache_proxy.LOGGER, "log") as log:
            self.proxy._seed_prefix(
                {"messages": [{"role": "system", "content": private_prompt}]},
                prefix_file,
                excluded_slot_id=1,
            )

        self.assertFalse(prefix_file.exists())
        self.assertFalse(manifest_file.exists())
        self.assertNotIn(private_prompt, str(log.call_args_list))
        self.assertIn("prefix_seed_token_count_mismatch", str(log.call_args_list))

    def test_prefix_seed_skips_tokens_already_published_in_valid_manifest(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(return_value=[{"id": 0, "is_processing": False}])
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3))
        existing = self._write_shared("local-llm-prefix-existing.bin", [1, 2, 3])
        requested = Path(self.tempdir.name, "local-llm-prefix-requested.bin")
        self.proxy._json_request = Mock()
        self.proxy._save = Mock()

        with patch.object(cache_proxy.LOGGER, "log") as log:
            self.proxy._seed_prefix(self.body, requested, excluded_slot_id=1)

        self.assertTrue(existing.exists())
        self.assertFalse(requested.exists())
        self.proxy._json_request.assert_not_called()
        self.proxy._save.assert_not_called()
        self.assertIn("prefix_seed_duplicate_skipped", str(log.call_args_list))

    def test_prefix_seed_stops_if_foreground_arrives_after_token_discovery(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(return_value=[{"id": 0, "is_processing": False}])
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3))
        self.proxy._has_foreground_waiters = Mock(side_effect=[False, True])
        self.proxy._json_request = Mock()
        self.proxy._save = Mock()

        self.proxy._seed_prefix(self.body, Path(self.tempdir.name, "local-llm-prefix-wait.bin"), 1)

        self.proxy._json_request.assert_not_called()
        self.proxy._save.assert_not_called()

    def test_manifest_publish_failure_preserves_old_valid_pair_and_removes_new_snapshot(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(return_value=[{"id": 0, "is_processing": False}])
        self.proxy._render_tokens = Mock(return_value=(4, 5, 6))
        self.proxy._json_request = Mock(return_value={})
        prefix_file = self._write_shared("local-llm-prefix-existing.bin", [1, 2, 3])
        manifest_file = Path(self.tempdir.name, manifest_filename(prefix_file.name))
        old_manifest = manifest_file.read_bytes()

        def save_new(_slot, target, **_kwargs):
            target.write_bytes(b"new snapshot")
            return 3

        self.proxy._save = Mock(side_effect=save_new)
        self.proxy._write_manifest = Mock(side_effect=OSError("disk full"))

        self.proxy._seed_prefix(self.body, prefix_file, excluded_slot_id=1)

        self.assertEqual(prefix_file.read_bytes(), b"snapshot")
        self.assertEqual(manifest_file.read_bytes(), old_manifest)
        snapshots = list(Path(self.tempdir.name).glob("local-llm-prefix-*.bin"))
        self.assertEqual(snapshots, [prefix_file])

    def test_write_manifest_requires_snapshot(self):
        with self.assertRaises(RuntimeError):
            self.proxy._write_manifest(Path(self.tempdir.name, "missing.bin"), (1, 2))

    def test_prefix_seed_never_borrows_clean_owned_slot_and_releases_foreground_lock(self):
        self.proxy.prefix_seed_delay_seconds = 0
        owner_file = Path(self.tempdir.name, "owner.bin")
        owner_file.write_bytes(b"persisted")
        self.proxy.session_states = {
            "owner": SlotState(0, "key", 10, owner_file, False),
        }
        self.proxy._slots = Mock(
            return_value=[{"id": 0, "is_processing": False, "n_prompt_tokens": 10}]
        )
        self.proxy._render_tokens = Mock(return_value=(1, 2, 3))
        prefix_file = Path(self.tempdir.name, "local-llm-prefix-golden.bin")
        pending = cache_proxy.PendingPrefixSeed(
            self.body,
            prefix_file,
            0,
            (1, 2, 3),
        )
        self.proxy.pending_prefix_seeds[prefix_file] = pending
        self.proxy.prefix_seeds_in_flight.add(prefix_file)
        seed_started = threading.Event()
        unblock_seed = threading.Event()

        def block_seed(*_args, **_kwargs):
            seed_started.set()
            unblock_seed.wait(1)
            return {}

        self.proxy._json_request = Mock(side_effect=block_seed)

        def save(slot, target, **_kwargs):
            target.write_bytes(b"snapshot")
            return 10 if target.name.endswith(".seed-owner.tmp") else 3

        self.proxy._save = Mock(side_effect=save)
        self.proxy._restore = Mock(return_value=10)
        worker = threading.Thread(
            target=self.proxy._seed_prefix,
            args=(self.body, prefix_file, 0, (1, 2, 3)),
        )

        worker.start()
        seed_started.wait(0.1)
        worker.join(0.1)
        worker_was_alive = worker.is_alive()
        foreground_acquired = self.proxy.operation_lock.acquire(timeout=0.05)
        if foreground_acquired:
            self.proxy.operation_lock.release()
        unblock_seed.set()
        worker.join(1)

        self.assertFalse(worker_was_alive)
        self.assertTrue(foreground_acquired)
        self.proxy._json_request.assert_not_called()
        self.proxy._save.assert_not_called()
        self.proxy._restore.assert_not_called()
        self.assertEqual(self.proxy.pending_prefix_seeds[prefix_file], pending)
        self.assertNotIn(prefix_file, self.proxy.prefix_seeds_in_flight)

    def test_one_slot_seed_skips_dirty_missing_mismatched_or_waiting_owner(self):
        self.proxy.prefix_seed_delay_seconds = 0
        owner_file = Path(self.tempdir.name, "owner.bin")
        owner_file.write_bytes(b"persisted")
        self.proxy._slots = Mock(return_value=[{"id": 0, "is_processing": False, "n_prompt_tokens": 10}])
        self.proxy._json_request = Mock()
        self.proxy._save = Mock()
        prefix = Path(self.tempdir.name, "local-llm-prefix-golden.bin")
        for state in (
            SlotState(0, "key", 10, owner_file, True),
            SlotState(0, "key", 10, Path(self.tempdir.name, "missing.bin"), False),
            SlotState(0, "key", 9, owner_file, False),
        ):
            with self.subTest(state=state):
                self.proxy.session_states = {"owner": state}
                self.proxy._seed_prefix(self.body, prefix, 0)
        self.proxy.foreground_waiters = 1
        self.proxy.session_states = {"owner": SlotState(0, "key", 10, owner_file, False)}
        self.proxy._seed_prefix(self.body, prefix, 0)

        self.proxy._json_request.assert_not_called()
        self.proxy._save.assert_not_called()

    def test_prefix_seed_skips_when_foreground_is_waiting_or_proxy_is_busy(self):
        self.proxy.prefix_seed_delay_seconds = 0
        prefix_file = Path(self.tempdir.name, "prefix.bin")
        self.proxy.foreground_waiters = 1
        self.proxy._json_request = Mock()
        self.proxy._seed_prefix({"messages": [{"role": "system", "content": "rules"}]}, prefix_file, 1)
        self.proxy._json_request.assert_not_called()

        self.proxy.foreground_waiters = 0
        busy_lock = Mock()
        busy_lock.acquire.return_value = False
        self.proxy.operation_lock = busy_lock
        self.proxy._seed_prefix({"messages": [{"role": "system", "content": "rules"}]}, prefix_file, 1)
        busy_lock.release.assert_not_called()

    def test_prefix_seed_contains_and_cleans_up_failures(self):
        self.proxy.prefix_seed_delay_seconds = 0
        prefix_file = Path(self.tempdir.name, "prefix.bin")
        self.proxy._slots = Mock(side_effect=RuntimeError("slots unavailable"))
        self.proxy._seed_prefix({"messages": [{"role": "system", "content": "rules"}]}, prefix_file, 1)
        self.assertNotIn(prefix_file, self.proxy.prefix_seeds_in_flight)

    def test_prefix_seed_payload_stops_before_user_message(self):
        body = {
            **self.body,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        seed = self.proxy._prefix_seed_payload(body)

        self.assertEqual(seed["messages"], [self.body["messages"][0]])
        self.assertEqual(seed["n_predict"], 0)
        self.assertFalse(seed["add_generation_prompt"])
        self.assertFalse(seed["cache_prompt"])

    def test_anonymous_project_affinity_follows_project_prefix_not_user_message(self):
        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        changed_project = {
            **self.body,
            "messages": [{"role": "system", "content": "other project"}, self.body["messages"][1]],
        }

        self.assertEqual(_anonymous_session_id(self.body), _anonymous_session_id(other_body))
        self.assertNotEqual(_anonymous_session_id(self.body), _anonymous_session_id(changed_project))

    def test_anonymous_conversation_affinity_uses_first_non_system_message(self):
        continued = {
            **self.body,
            "messages": self.body["messages"] + [
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ],
        }
        different_start = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "different task"}],
        }

        original = _anonymous_session_id(self.body, "conversation-start")
        self.assertEqual(original, _anonymous_session_id(continued, "conversation-start"))
        self.assertNotEqual(original, _anonymous_session_id(different_start, "conversation-start"))

    def test_affinity_helpers_use_priority_and_strip_proxy_fields(self):
        handler = RecordingHandler(
            headers={
                "X-Session-Affinity": "  header-session  ",
                "X-Pi-Session-Id": "fallback-session",
            }
        )
        self.assertEqual(_session_id(handler), "header-session")
        self.assertIsNone(_session_id(RecordingHandler()))
        self.assertEqual(_body_session_id({"session_id": " body-session "}), "body-session")
        self.assertEqual(_body_session_id({"prompt_cache_key": " prompt-key "}), "prompt-key")
        self.assertIsNone(_body_session_id({"session_id": 42, "prompt_cache_key": ""}))

        conversation_handler = RecordingHandler(headers={"X-Conversation-Id": "conversation-header"})
        self.assertEqual(_session_id(conversation_handler), "conversation-header")
        self.assertEqual(_body_session_id({"conversation_id": "conversation-body"}), "conversation-body")
        self.assertEqual(
            _body_session_id({"extra_body": {"session_id": " nested-session "}}),
            "nested-session",
        )

        body = {
            "session_id": "session",
            "conversation_id": "conversation",
            "prompt_cache_key": "key",
            "extra_body": {"session_id": "nested", "keep": "upstream-extension"},
            "messages": [],
        }
        stripped = _without_proxy_affinity_fields(body)
        self.assertEqual(
            stripped,
            {"extra_body": {"keep": "upstream-extension"}, "messages": []},
        )
        self.assertEqual(body["session_id"], "session")
        self.assertEqual(
            _without_proxy_affinity_fields({"extra_body": {"conversation_id": "nested"}}),
            {},
        )

    def test_media_detection_covers_images_and_message_content(self):
        self.assertTrue(_has_media({"images": ["image-data"]}))
        self.assertTrue(
            _has_media(
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url"}]},
                    ]
                }
            )
        )
        self.assertFalse(
            _has_media(
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    ]
                }
            )
        )
        self.assertFalse(_has_media({"messages": [{"role": "user", "content": "hello"}]}))

    def test_wait_for_idle_slot_prefers_unowned_and_times_out(self):
        self.proxy.session_states["session-a"] = SlotState(1, cache_key(self.body), 10)
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 3},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 1},
        ]
        with patch.object(self.proxy, "_slots", return_value=slots):
            self.assertEqual(self.proxy._wait_for_idle_slot(), 0)

        self.proxy.wait_seconds = 0
        with patch.object(self.proxy, "_slots", return_value=[]):
            with self.assertRaises(TimeoutError):
                self.proxy._wait_for_idle_slot()

        self.proxy.wait_seconds = 1
        with patch.object(self.proxy, "_slots", return_value=[]), patch(
            "cache_proxy.time.monotonic", side_effect=[0, 0, 2]
        ), patch("cache_proxy.time.sleep") as sleep:
            with self.assertRaises(TimeoutError):
                self.proxy._wait_for_idle_slot()
        sleep.assert_called_once_with(0.5)

    def test_forward_streams_body_and_filters_hop_by_hop_headers(self):
        response = FakeResponse(
            status=201,
            headers=[("Content-Type", "text/plain"), ("Content-Length", "2"), ("Connection", "close")],
            chunks=[b"o", b"k"],
        )
        connection = FakeConnection("host", 80, response=response)
        handler = RecordingHandler(
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer token",
                "Accept": "text/event-stream",
                "User-Agent": "test",
            }
        )

        with patch("cache_proxy.HTTPConnection", return_value=connection):
            result = self.proxy.forward(handler, "POST", "/v1/test", b"body")

        self.assertEqual(result.status, 201)
        self.assertEqual(connection.requests[0][0:2], ("POST", "/v1/test"))
        self.assertEqual(connection.requests[0][2], b"body")
        self.assertIn(("Content-Type", "text/plain"), handler.sent_headers)
        self.assertNotIn(("Connection", "close"), handler.sent_headers)
        self.assertNotIn(("Content-Length", "2"), handler.sent_headers)
        self.assertIn(("Transfer-Encoding", "chunked"), handler.sent_headers)
        self.assertEqual(b"".join(handler.wfile.data), b"1\r\no\r\n1\r\nk\r\n0\r\n\r\n")
        self.assertEqual(response.read_calls, 0)
        self.assertEqual(response.read1_calls, 3)
        self.assertTrue(connection.closed)

    def test_forward_head_preserves_content_length_without_writing_a_body(self):
        response = FakeResponse(
            status=200,
            headers=[("Content-Type", "application/json"), ("Content-Length", "2")],
            chunks=[b"{}"],
        )
        connection = FakeConnection("host", 80, response=response)
        handler = RecordingHandler()

        with patch("cache_proxy.HTTPConnection", return_value=connection):
            result = self.proxy.forward(handler, "HEAD", "/props", b"")

        self.assertEqual(result.status, 200)
        self.assertIn(("Content-Length", "2"), handler.sent_headers)
        self.assertNotIn(("Transfer-Encoding", "chunked"), handler.sent_headers)
        self.assertEqual(handler.wfile.data, [])
        self.assertTrue(connection.closed)

    def test_forward_bounds_completion_metadata_buffer(self):
        oversized = b"x" * (MAX_METADATA_BYTES + 1)
        connection = FakeConnection(
            "host",
            80,
            response=FakeResponse(status=200, chunks=[oversized]),
        )
        handler = RecordingHandler()

        with patch("cache_proxy.HTTPConnection", return_value=connection):
            result = self.proxy.forward(handler, "POST", "/v1/chat/completions", b"{}")

        self.assertEqual(result, ForwardResult(200))
        self.assertTrue(connection.closed)

    def test_forward_returns_status_when_client_disconnects(self):
        connection = FakeConnection("host", 80, response=FakeResponse(status=200, chunks=[b"ok"]))
        handler = RecordingHandler(broken=True)

        with patch("cache_proxy.HTTPConnection", return_value=connection):
            result = self.proxy.forward(handler, "GET", "/health", b"")

        self.assertEqual(result.status, 200)
        self.assertTrue(connection.closed)

    def test_completion_metadata_parses_json_and_split_sse_chunks(self):
        regular = _completion_metadata(
            [b'{"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens_details":{"cached_tokens":42}}}']
        )
        self.assertEqual(regular, CompletionMetadata("stop", 42))

        streamed = _completion_metadata(
            [
                b'data: {"choices":[{"delta":{"content":"x"},"finish_',
                b'reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],',
                b'"usage":{"prompt_tokens_details":{"cached_tokens":17}}}\n\ndata: [DONE]\n\n',
            ]
        )
        self.assertEqual(streamed, CompletionMetadata("tool_calls", 17))

    def test_completion_metadata_tolerates_non_completion_and_malformed_sse(self):
        self.assertEqual(_metadata_from_object([]), CompletionMetadata())
        self.assertEqual(
            _metadata_from_object(
                {
                    "choices": [None, {"finish_reason": None}],
                    "usage": {"prompt_tokens_details": {"cached_tokens": "not-an-int"}},
                }
            ),
            CompletionMetadata(),
        )
        self.assertEqual(_metadata_from_object({"usage": "unknown"}), CompletionMetadata())
        self.assertEqual(_metadata_from_object({"choices": 1}), CompletionMetadata())
        self.assertEqual(
            _completion_metadata(
                [b"not-json\n\ndata:\n\ndata: [DONE]\n\ndata: {broken}\n\ndata: []\n\n"]
            ),
            CompletionMetadata(),
        )

    def test_boolean_environment_parser_is_strict(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_env_bool("TEST_BOOLEAN", True))
        for value in ("1", "true", "YES", " on "):
            with patch.dict(os.environ, {"TEST_BOOLEAN": value}):
                self.assertTrue(_env_bool("TEST_BOOLEAN", False))
        for value in ("0", "false", "NO", " off "):
            with patch.dict(os.environ, {"TEST_BOOLEAN": value}):
                self.assertFalse(_env_bool("TEST_BOOLEAN", True))
        with patch.dict(os.environ, {"TEST_BOOLEAN": "maybe"}), self.assertRaises(ValueError):
            _env_bool("TEST_BOOLEAN", True)

    def test_forward_headers_preserve_supported_request_headers(self):
        handler = RecordingHandler(
            headers={
                "Content-Type": "application/custom",
                "Authorization": "token",
                "Accept": "application/json",
                "User-Agent": "agent",
                "Origin": "https://client.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
                "X-Api-Key": "anthropic-token",
                "Anthropic-Version": "2023-06-01",
                "Anthropic-Beta": "test-beta",
                "OpenAI-Organization": "org-test",
                "OpenAI-Project": "project-test",
                "Idempotency-Key": "request-test",
            }
        )

        headers = self.proxy._forward_headers(handler)

        self.assertEqual(
            headers,
            {
                "Content-Type": "application/custom",
                "Authorization": "token",
                "Accept": "application/json",
                "User-Agent": "agent",
                "Origin": "https://client.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
                "X-Api-Key": "anthropic-token",
                "Anthropic-Version": "2023-06-01",
                "Anthropic-Beta": "test-beta",
                "OpenAI-Organization": "org-test",
                "OpenAI-Project": "project-test",
                "Idempotency-Key": "request-test",
            },
        )

    def test_json_request_and_slots_handle_success_empty_and_error(self):
        success = FakeConnection("host", 80, response=FakeResponse(raw=b'{"ok": true}'))
        with patch("cache_proxy.HTTPConnection", return_value=success):
            self.assertEqual(self.proxy._json_request("POST", "/test", {"x": 1}), {"ok": True})
        self.assertTrue(success.closed)

        empty = FakeConnection("host", 80, response=FakeResponse(raw=b""))
        with patch("cache_proxy.HTTPConnection", return_value=empty):
            self.assertEqual(self.proxy._json_request("POST", "/test", {}), {})

        failed = FakeConnection(
            "host", 80, response=FakeResponse(status=500, raw=b"SECRET_PROMPT_CONTENT")
        )
        with patch("cache_proxy.HTTPConnection", return_value=failed):
            with self.assertRaises(RuntimeError) as raised:
                self.proxy._json_request("POST", "/test", {})
        self.assertEqual(str(raised.exception), "llama POST /test returned 500")
        self.assertNotIn("SECRET_PROMPT_CONTENT", str(raised.exception))
        self.assertTrue(failed.closed)

        slots = FakeConnection("host", 80, response=FakeResponse(raw=b'[{"id": 0}]'))
        with patch("cache_proxy.HTTPConnection", return_value=slots):
            self.assertEqual(self.proxy._slots(), [{"id": 0}])

        non_list = FakeConnection("host", 80, response=FakeResponse(raw=b"{}"))
        with patch("cache_proxy.HTTPConnection", return_value=non_list):
            self.assertEqual(self.proxy._slots(), [])

        failed_slots = FakeConnection("host", 80, response=FakeResponse(status=503))
        with patch("cache_proxy.HTTPConnection", return_value=failed_slots):
            with self.assertRaises(RuntimeError):
                self.proxy._slots()

    def test_save_rejects_zero_tokens(self):
        self.proxy._json_request = Mock(return_value={"n_saved": 0})

        with self.assertRaises(RuntimeError):
            self.proxy._save(0, Path(self.tempdir.name, "session.bin"))

    def test_save_removes_partial_temporary_snapshot_after_transport_failure(self):
        target = Path(self.tempdir.name, "session.bin")

        def fail_after_writing(_method, _path, payload):
            Path(self.tempdir.name, payload["filename"]).write_bytes(b"partial")
            raise TimeoutError("timeout")

        self.proxy._json_request = Mock(side_effect=fail_after_writing)

        with self.assertRaises(TimeoutError):
            self.proxy._save(0, target)

        self.assertFalse(target.with_suffix(".bin.tmp").exists())

    def test_save_logs_temporary_cleanup_failure_without_hiding_save_error(self):
        self.proxy._json_request = Mock(return_value={"n_saved": 0})

        with (
            patch.object(Path, "unlink", side_effect=[None, PermissionError("denied")]),
            patch.object(cache_proxy.LOGGER, "log") as logger,
            self.assertRaises(RuntimeError),
        ):
            self.proxy._save(0, Path(self.tempdir.name, "session.bin"))

        self.assertIn("snapshot_temp_cleanup_failed", logger.call_args.args[2])

    def test_proxy_handler_get_and_non_chat_post_forward_successfully(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        proxy = CapturingProxy()
        ProxyHandler.proxy = proxy
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
            self.assertEqual(proxy.forwarded[-1][1], "/health")

            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("GET", "/v1/props?format=json")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
            self.assertEqual(proxy.forwarded[-1][1], "/props?format=json")

            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("HEAD", "/v1/props")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
            self.assertEqual(proxy.forwarded[-1][0:2], ("HEAD", "/props"))

            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("OPTIONS", "/v1/models")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
            self.assertEqual(proxy.forwarded[-1][0:2], ("OPTIONS", "/v1/models"))

            for requested_path, upstream_path in (
                ("/v1/tokenize", "/tokenize"),
                ("/v1/detokenize", "/detokenize"),
                ("/v1/apply-template", "/apply-template"),
                ("/v1/responses/input_tokens", "/v1/responses/input_tokens"),
                ("/v1/chat/completions/input_tokens", "/v1/chat/completions/input_tokens"),
                ("/v1/messages/count_tokens", "/v1/messages/count_tokens"),
                ("/v1/chat/completions/control", "/v1/chat/completions/control"),
            ):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request(
                    "POST",
                    requested_path,
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
                connection.close()
                self.assertEqual(proxy.forwarded[-1][1], upstream_path)
            self.assertEqual(proxy.uncached_reasons, [])

            unmanaged_paths = (
                "/completion",
                "/v1/completions",
                "/infill",
                "/embedding",
                "/embeddings",
                "/v1/embeddings",
                "/rerank",
                "/reranking",
                "/v1/rerank",
                "/v1/responses",
                "/v1/messages",
            )
            for unmanaged_path in unmanaged_paths:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request(
                    "POST",
                    unmanaged_path,
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
                connection.close()
                self.assertEqual(proxy.forwarded[-1][1], unmanaged_path)
            self.assertEqual(
                proxy.uncached_reasons, ["before_unmanaged_post"] * len(unmanaged_paths)
            )

            forwarded_before = list(proxy.forwarded)
            for method, path in (
                ("GET", "/slots"),
                ("HEAD", "/slots"),
                ("OPTIONS", "/slots/0"),
                ("POST", "/slots/0?action=save"),
                ("GET", "/tools"),
                ("GET", "/lora-adapters"),
                ("POST", "/props"),
                ("POST", "/lora-adapters"),
                ("POST", "/models"),
                ("POST", "/models/load"),
                ("POST", "/models/unload"),
                ("DELETE", "/models?model=test"),
                ("DELETE", "/future-resource"),
                ("POST", "/future-admin-route"),
            ):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request(method, path, body=b"{}" if method == "POST" else None)
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 404)
                connection.close()
            self.assertEqual(proxy.forwarded, forwarded_before)

            for content_length in ("not-a-number", "-1"):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request(
                    "POST",
                    "/v1/completions",
                    headers={"Content-Length": content_length},
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 400)
                connection.close()
            self.assertEqual(proxy.forwarded, forwarded_before)
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_get_and_non_chat_post_return_503_on_upstream_error(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = FailingForwardProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for method, path in (
                ("GET", "/health"),
                ("HEAD", "/health"),
                ("OPTIONS", "/v1/models"),
                ("POST", "/v1/completions"),
                ("POST", "/v1/tokenize"),
            ):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                body = b"{}" if method == "POST" else None
                headers = {"Content-Type": "application/json"} if body is not None else {}
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 503)
                connection.close()
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_rejects_invalid_json_and_forwards_media(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        proxy = CapturingProxy()
        proxy.require_session_id = True
        ProxyHandler.proxy = proxy
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("POST", "/v1/chat/completions", body=b"not-json")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 400)
            connection.close()

            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
            self.assertEqual(proxy.forwarded[-1][0], "POST")
            self.assertEqual(proxy.uncached_reasons, ["before_media"])
            self.assertEqual(proxy.session_ids, [])
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_returns_503_for_media_upstream_error(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = FailingForwardProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"images": ["image-data"]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 503)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_contains_snapshot_finalization_error(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = FinishFailingProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_error_response_does_not_write_second_response_after_disconnect(self):
        handler = RecordingHandler()
        handler._proxy_response_started = False
        ProxyHandler._send_upstream_error(handler, RuntimeError("downstream failed"))
        self.assertEqual(handler.errors[0][0], 503)

        started = RecordingHandler()
        started._proxy_response_started = True
        ProxyHandler._send_upstream_error(started, RuntimeError("downstream failed"))
        self.assertEqual(started.errors, [])

    def test_proxy_handler_log_message_uses_client_address(self):
        handler = RecordingHandler()
        handler.address_string = lambda: "127.0.0.1"
        with patch.object(cache_proxy.LOGGER, "info") as info:
            ProxyHandler.log_message(handler, "status %s", 200)
        info.assert_called_once_with("%s - %s", "127.0.0.1", "status 200")

    def test_main_builds_server_from_environment_and_closes_on_interrupt(self):
        servers = []

        class MainServer:
            def __init__(self, address, handler):
                self.address = address
                self.handler = handler
                self.closed = False
                servers.append(self)

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                self.closed = True

        env = {
            "PI_LLAMA_UPSTREAM": "http://127.0.0.1:9999/api",
            "PI_LLAMA_CACHE_HOST": "127.0.0.1",
            "PI_LLAMA_CACHE_PORT": "19082",
            "PI_LLAMA_CACHE_DIR": self.tempdir.name,
            "PI_LLAMA_CACHE_MAX_GIB": "1",
            "PI_LLAMA_CACHE_WAIT_SECONDS": "1",
            "PI_LLAMA_CACHE_PREFIX_SEED_DELAY": "0",
            "PI_LLAMA_CACHE_PREFIX_SEED_TIMEOUT": "321",
        }
        with patch.dict(os.environ, env, clear=False), patch("http.server.ThreadingHTTPServer", MainServer):
            runpy.run_path(cache_proxy.__file__, run_name="__main__")

        self.assertEqual(servers[0].address, ("127.0.0.1", 19082))
        self.assertEqual(servers[0].handler.proxy.prefix_seed_timeout_seconds, 321.0)
        self.assertTrue(servers[0].closed)

    def test_main_handles_sigterm_and_logs_shutdown_flush_failure(self):
        servers = []

        class SigtermServer:
            def __init__(self, address, handler):
                self.address = address
                self.handler = handler
                self.closed = False
                servers.append(self)

            def serve_forever(self):
                self.handler.proxy.flush_dirty = Mock(side_effect=RuntimeError("disk unavailable"))
                signal.raise_signal(signal.SIGTERM)

            def server_close(self):
                self.closed = True

        env = {
            "PI_LLAMA_CACHE_HOST": "127.0.0.1",
            "PI_LLAMA_CACHE_PORT": "19083",
            "PI_LLAMA_CACHE_DIR": self.tempdir.name,
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch("http.server.ThreadingHTTPServer", SigtermServer),
            patch.object(cache_proxy.LOGGER, "exception") as exception,
        ):
            runpy.run_path(cache_proxy.__file__, run_name="__main__")

        self.assertTrue(servers[0].closed)
        exception.assert_called_once_with("failed to flush dirty snapshots during shutdown")

    def test_upstream_unavailable_returns_503(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = UnavailableProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "rules"},
                            {"role": "user", "content": "hello"},
                        ]
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 503)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_prune_removes_local_llm_snapshots_over_limit(self):
        proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=0,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        snapshot = Path(self.tempdir.name, "local-llm-session-old.bin")
        snapshot.write_bytes(b"snapshot")

        proxy._prune()

        self.assertFalse(snapshot.exists())

    def test_completion_prunes_without_deleting_another_active_snapshot(self):
        active_snapshot = Path(self.tempdir.name, "local-llm-session-active.bin")
        active_snapshot.write_bytes(b"durable-active-checkpoint")
        completed_snapshot = Path(self.tempdir.name, "local-llm-session-completed.bin")
        self.proxy.max_cache_bytes = 0
        self.proxy.active_slot_ids.update({0, 1})
        self.proxy.active_slot_snapshots.update({0: active_snapshot, 1: completed_snapshot})
        self.proxy._save = Mock(
            side_effect=lambda _slot, target, **_kwargs: (
                target.write_bytes(b"completed-checkpoint"),
                10,
            )[1]
        )
        plan = SnapshotPlan(
            "completed-session",
            1,
            "completed-prefix",
            completed_snapshot,
            Path(self.tempdir.name, "local-llm-prefix-completed.bin"),
            {"messages": [], "tools": []},
            True,
        )

        self.proxy.complete_request(plan, 200)

        self.assertTrue(active_snapshot.exists())
        self.assertFalse(completed_snapshot.exists())
        self.assertEqual(self.proxy.active_slot_ids, {0})
        self.assertEqual(self.proxy.active_slot_snapshots, {0: active_snapshot})

    def test_repeated_overlapping_completions_enforce_limit_except_active_snapshot(self):
        active_snapshot = Path(self.tempdir.name, "local-llm-session-active.bin")
        active_snapshot.write_bytes(b"durable-active-checkpoint")
        self.proxy.max_cache_bytes = active_snapshot.stat().st_size
        self.proxy.active_slot_ids.add(0)
        self.proxy.active_slot_snapshots[0] = active_snapshot
        self.proxy._save = Mock(
            side_effect=lambda _slot, target, **_kwargs: (
                target.write_bytes(b"completed-checkpoint"),
                10,
            )[1]
        )

        for index in range(5):
            completed_snapshot = Path(
                self.tempdir.name,
                f"local-llm-session-completed-{index}.bin",
            )
            self.proxy.active_slot_ids.add(1)
            self.proxy.active_slot_snapshots[1] = completed_snapshot
            plan = SnapshotPlan(
                f"completed-session-{index}",
                1,
                f"completed-prefix-{index}",
                completed_snapshot,
                Path(self.tempdir.name, f"local-llm-prefix-completed-{index}.bin"),
                {"messages": [], "tools": []},
                True,
            )

            self.proxy.complete_request(plan, 200)

            self.assertTrue(active_snapshot.exists())
            self.assertFalse(completed_snapshot.exists())
            self.assertEqual(self.proxy.active_slot_ids, {0})
            self.assertEqual(self.proxy.active_slot_snapshots, {0: active_snapshot})
            self.assertLessEqual(
                sum(path.stat().st_size for path in Path(self.tempdir.name).glob("local-llm-*.bin")),
                self.proxy.max_cache_bytes,
            )

    def test_stale_cancel_does_not_release_reused_slot_after_completion(self):
        first = Path(self.tempdir.name, "local-llm-session-first.bin")
        second = Path(self.tempdir.name, "local-llm-session-second.bin")
        first_plan = SnapshotPlan(
            "first",
            0,
            "first-prefix",
            first,
            Path(self.tempdir.name, "local-llm-prefix-first.bin"),
            {"messages": [], "tools": []},
            True,
        )
        self.proxy.active_slot_ids.add(0)
        self.proxy.active_slot_snapshots[0] = first
        self.proxy._save = Mock(
            side_effect=lambda _slot, target, **_kwargs: (target.write_bytes(b"snapshot"), 10)[1]
        )

        self.proxy.complete_request(first_plan, 200)
        self.proxy.active_slot_ids.add(0)
        self.proxy.active_slot_snapshots[0] = second

        self.proxy.cancel_request(first_plan)

        self.assertEqual(self.proxy.active_slot_snapshots[0], second)
        self.assertEqual(self.proxy.active_slot_ids, {0})

    def test_direct_finish_still_enforces_cache_limit(self):
        old_snapshot = Path(self.tempdir.name, "local-llm-session-old-direct.bin")
        old_snapshot.write_bytes(b"old")
        new_snapshot = Path(self.tempdir.name, "local-llm-session-new-direct.bin")
        self.proxy.max_cache_bytes = len(b"new")
        self.proxy._save = Mock(
            side_effect=lambda _slot, target, **_kwargs: (target.write_bytes(b"new"), 10)[1]
        )
        plan = SnapshotPlan(
            "direct",
            0,
            "direct-prefix",
            new_snapshot,
            Path(self.tempdir.name, "local-llm-prefix-direct.bin"),
            {"messages": [], "tools": []},
            True,
        )

        self.proxy.finish(plan, 200)

        self.assertFalse(old_snapshot.exists())
        self.assertTrue(new_snapshot.exists())

    def test_prune_removes_prefix_manifest_with_snapshot(self):
        snapshot = self._write_shared("local-llm-prefix-old.bin", [1, 2])
        manifest = Path(self.tempdir.name, manifest_filename(snapshot.name))
        self.proxy.max_cache_bytes = 0

        self.proxy._prune()

        self.assertFalse(snapshot.exists())
        self.assertFalse(manifest.exists())

    def test_prune_removes_orphan_manifest_and_temporary_metadata(self):
        orphan = Path(
            self.tempdir.name,
            "local-llm-prefix-orphan.bin.manifest.json",
        )
        temporary_snapshot = Path(self.tempdir.name, "local-llm-session-save.bin.tmp")
        temporary_manifest = Path(
            self.tempdir.name,
            "local-llm-prefix-seed.bin.manifest.json.tmp",
        )
        orphan.write_text("{}")
        temporary_snapshot.write_bytes(b"partial")
        temporary_manifest.write_text("partial")

        self.proxy._prune()

        self.assertFalse(orphan.exists())
        self.assertFalse(temporary_snapshot.exists())
        self.assertFalse(temporary_manifest.exists())

    def test_restore_refreshes_snapshot_lru_timestamp(self):
        snapshot = Path(self.tempdir.name, "local-llm-session-old.bin")
        snapshot.write_bytes(b"snapshot")
        os.utime(snapshot, (1, 1))

        self.proxy._restore_first_available(0, (snapshot,))

        self.assertGreater(snapshot.stat().st_mtime, 1)

    def test_snapshot_touch_is_best_effort(self):
        missing = Path(self.tempdir.name, "missing.bin")
        self.assertIsNone(self.proxy._touch_snapshot(missing))

        snapshot = Path(self.tempdir.name, "snapshot.bin")
        snapshot.write_bytes(b"snapshot")
        with (
            patch.object(Path, "touch", side_effect=PermissionError("read-only")),
            patch.object(cache_proxy.LOGGER, "log") as log,
        ):
            self.assertIsNone(self.proxy._touch_snapshot(snapshot))
        self.assertIn("snapshot_touch_failed", log.call_args.args[2])

    def test_uncached_request_flushes_dirty_state_and_resets_ownership(self):
        target = Path(self.tempdir.name, "dirty-session.bin")
        self.proxy.session_states = {
            "session-a": SlotState(0, cache_key(self.body), 12, target, True),
        }

        self.proxy.prepare_uncached("before_media")

        self.assertEqual(self.proxy.session_states, {})
        self.assertTrue(target.exists())
        self.assertEqual(FakeLlamaHandler.saved[-1], target.name + ".tmp")

    def test_body_session_id_is_used_for_affinity(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        capturing_proxy = CapturingProxy()
        ProxyHandler.proxy = capturing_proxy
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "session_id": "body-session",
                        "messages": [
                            {"role": "system", "content": "rules"},
                            {"role": "user", "content": "hello"},
                        ],
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(capturing_proxy.session_ids, ["body-session"])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_required_session_id_rejects_anonymous_chat_request(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        capturing_proxy = CapturingProxy()
        capturing_proxy.require_session_id = True
        ProxyHandler.proxy = capturing_proxy
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 400)
            self.assertEqual(capturing_proxy.session_ids, [])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_structured_events_never_log_raw_session_id(self):
        raw_session_id = "private-session-that-must-not-appear"
        with patch.object(cache_proxy.LOGGER, "log") as log:
            _, plan = self.proxy.prepare(self.body, raw_session_id)
            self.proxy.finish(plan, 200)

        rendered = " ".join(str(call) for call in log.call_args_list)
        self.assertNotIn(raw_session_id, rendered)
        self.assertIn(cache_proxy._session_ref(raw_session_id), rendered)

    def test_non_object_json_returns_400(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = UnavailableProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body="[]",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 400)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy


class KvControlParsingTests(unittest.TestCase):
    def test_tag_is_stripped_from_system_and_first_message_only(self):
        body = {
            "messages": [
                {"role": "system", "content": "rules [[x-kv-save: false]] more"},
                {"role": "user", "content": "[[x-kv-save-growth: 4096]] question"},
                {"role": "user", "content": "later [[x-kv-restore: false]] message"},
            ]
        }
        found = cache_proxy._extract_kv_tags(body, "")
        self.assertEqual(found, {"save": False, "save-growth": 4096})
        self.assertEqual(body["messages"][0]["content"], "rules  more")
        self.assertEqual(body["messages"][1]["content"], " question")
        self.assertEqual(body["messages"][2]["content"], "later [[x-kv-restore: false]] message")

    def test_tag_in_text_part_of_content_list(self):
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "a [[x-kv-restore: false]] b"}],
                }
            ]
        }
        found = cache_proxy._extract_kv_tags(body, "")
        self.assertEqual(found, {"restore": False})
        self.assertEqual(body["messages"][0]["content"][0]["text"], "a  b")

    def test_inline_prefix_gates_tag_recognition(self):
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": "[[x-kv-save: false]] [[b7-x-kv-save: false]]",
                }
            ]
        }
        found = cache_proxy._extract_kv_tags(body, "b7-")
        self.assertEqual(found, {"save": False})
        self.assertEqual(body["messages"][0]["content"], "[[x-kv-save: false]] ")

    def test_unknown_tag_key_is_stripped_and_invalid_value_ignored(self):
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": "[[x-kv-frobnicate: 1]] [[x-kv-save-growth: lots]]",
                }
            ]
        }
        with patch.object(cache_proxy.LOGGER, "log") as log:
            found = cache_proxy._extract_kv_tags(body, "")
        self.assertEqual(found, {})
        self.assertEqual(body["messages"][0]["content"], " ")
        rendered = str(log.call_args_list)
        self.assertIn("kv_tag_unknown", rendered)
        self.assertIn("kv_control_invalid", rendered)

    def test_body_fields_are_collected_and_stripped(self):
        body = {
            "kv_save": "false",
            "kv_restore_tokens": 256,
            "extra_body": {"kv_save_growth": "1024"},
            "messages": [],
        }
        found = cache_proxy._extract_kv_body_fields(body)
        self.assertEqual(found, {"save": False, "restore-tokens": 256, "save-growth": 1024})
        self.assertNotIn("kv_save", body)
        self.assertNotIn("kv_restore_tokens", body)
        self.assertNotIn("extra_body", body)

    def test_headers_are_collected(self):
        handler = RecordingHandler(
            headers={"X-KV-Save": "off", "X-KV-Restore-Tokens": "128", "X-KV-Save-Growth": "bad"}
        )
        found = cache_proxy._extract_kv_headers(handler.headers)
        self.assertEqual(found, {"save": False, "restore-tokens": 128})

    def test_resolution_precedence_header_body_tag_env(self):
        defaults = cache_proxy.KvControls(save_growth=10)
        resolved = cache_proxy._resolve_kv_controls(
            defaults,
            {"save": True},
            {"save": False, "restore": False},
            {"save": True, "restore": True, "restore-tokens": 7},
        )
        self.assertTrue(resolved.save)
        self.assertFalse(resolved.restore)
        self.assertEqual(resolved.restore_tokens, 7)
        self.assertEqual(resolved.save_growth, 10)

    def test_negative_thresholds_are_rejected(self):
        for kwargs in ({"save_growth": -1}, {"save_interval": -1}, {"restore_tokens": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LlamaCacheProxy(
                    upstream="http://127.0.0.1:1",
                    cache_dir=tempfile.mkdtemp(),
                    kv_controls=cache_proxy.KvControls(**kwargs),
                )


class KvControlsProxyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        FakeLlamaHandler.cache_dir = self.tempdir.name
        FakeLlamaHandler.restored = []
        FakeLlamaHandler.saved = []
        FakeLlamaHandler.restore_failed = False
        FakeLlamaHandler.restore_tokens = 10
        FakeLlamaHandler.slot_tokens = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.proxy = self._proxy()
        self.body = {
            "model": "qwen3.8:27b",
            "messages": [
                {"role": "system", "content": "project rules"},
                {"role": "user", "content": "request"},
            ],
            "tools": [],
        }

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tempdir.cleanup()

    def _proxy(self, **kwargs):
        return LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=1,
            wait_seconds=1,
            enable_prefix_seeding=False,
            **kwargs,
        )

    def test_save_disabled_skips_save_and_session_state(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200, controls=cache_proxy.KvControls(save=False))

        self.assertEqual(FakeLlamaHandler.saved, [])
        self.assertNotIn("session-a", self.proxy.session_states)

    def test_save_disabled_flushes_deferred_state_before_request(self):
        terminal = self._proxy(save_policy="terminal")
        _, plan = terminal.prepare(self.body, "session-a")
        FakeLlamaHandler.slot_tokens = 10
        terminal.finish(plan, 200, finish_reason="tool_calls")
        self.assertTrue(terminal.session_states["session-a"].dirty)
        saved_before = len(FakeLlamaHandler.saved)

        terminal.prepare(self.body, "session-a", controls=cache_proxy.KvControls(save=False))

        self.assertEqual(len(FakeLlamaHandler.saved), saved_before + 1)
        self.assertFalse(terminal.session_states["session-a"].dirty)

    def test_restore_disabled_starts_cold_despite_snapshot(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        self.assertEqual(len(FakeLlamaHandler.saved), 1)

        cold = self._proxy()
        _, cold_plan = cold.prepare(
            self.body, "session-a", controls=cache_proxy.KvControls(restore=False)
        )

        self.assertEqual(FakeLlamaHandler.restored, [])
        self.assertIsNone(cold_plan.slot_id)

    def test_save_growth_skips_small_rewrite_and_flush_persists_it(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        state = self.proxy.session_states["session-a"]
        self.assertEqual(state.saved_n_tokens, 10)

        _, hot_plan = self.proxy.prepare(self.body, "session-a")
        self.assertEqual(hot_plan.slot_id, 0)
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(hot_plan, 200, controls=cache_proxy.KvControls(save_growth=4096))

        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        state = self.proxy.session_states["session-a"]
        self.assertTrue(state.dirty)
        self.assertEqual(state.n_tokens, 12)
        self.assertEqual(state.saved_n_tokens, 10)

        self.proxy.flush_dirty("test")
        self.assertEqual(len(FakeLlamaHandler.saved), 2)
        self.assertFalse(self.proxy.session_states["session-a"].dirty)

    def test_save_growth_reached_saves_normally(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)

        _, hot_plan = self.proxy.prepare(self.body, "session-a")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(hot_plan, 200, controls=cache_proxy.KvControls(save_growth=2))

        self.assertEqual(len(FakeLlamaHandler.saved), 2)
        self.assertFalse(self.proxy.session_states["session-a"].dirty)

    def test_save_interval_skips_recent_rewrite_even_with_large_growth(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        self.assertIsNotNone(self.proxy.session_states["session-a"].saved_at)

        _, hot_plan = self.proxy.prepare(self.body, "session-a")
        FakeLlamaHandler.slot_tokens = 5000
        self.proxy.finish(hot_plan, 200, controls=cache_proxy.KvControls(save_interval=3600))

        self.assertEqual(len(FakeLlamaHandler.saved), 1)
        state = self.proxy.session_states["session-a"]
        self.assertTrue(state.dirty)
        self.assertEqual(state.saved_n_tokens, 10)

        self.proxy.flush_dirty("test")
        self.assertEqual(len(FakeLlamaHandler.saved), 2)
        state = self.proxy.session_states["session-a"]
        self.assertFalse(state.dirty)
        self.assertIsNotNone(state.saved_at)

    def test_save_interval_elapsed_saves_normally(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        state = self.proxy.session_states["session-a"]
        self.proxy.session_states["session-a"] = replace(state, saved_at=state.saved_at - 120)

        _, hot_plan = self.proxy.prepare(self.body, "session-a")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(hot_plan, 200, controls=cache_proxy.KvControls(save_interval=60))

        self.assertEqual(len(FakeLlamaHandler.saved), 2)
        self.assertFalse(self.proxy.session_states["session-a"].dirty)

    def test_save_interval_without_recorded_save_time_saves(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        state = self.proxy.session_states["session-a"]
        # A state without a recorded save time (e.g. carried over a restart)
        # must fail safe and save.
        self.proxy.session_states["session-a"] = replace(state, saved_at=None)

        _, hot_plan = self.proxy.prepare(self.body, "session-a")
        FakeLlamaHandler.slot_tokens = 12
        self.proxy.finish(hot_plan, 200, controls=cache_proxy.KvControls(save_interval=3600))

        self.assertEqual(len(FakeLlamaHandler.saved), 2)

    def test_restore_tokens_threshold_skips_known_small_session_snapshot(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        # Invalidate the hot slot so the next prepare takes the cold path.
        FakeLlamaHandler.slot_tokens = 99

        _, cold_plan = self.proxy.prepare(
            self.body, "session-a", controls=cache_proxy.KvControls(restore_tokens=100)
        )

        self.assertEqual(FakeLlamaHandler.restored, [])
        self.assertIsNone(cold_plan.slot_id)

    def test_handler_applies_tag_controls_end_to_end(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = self.proxy
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            body = {
                "model": "qwen3.8:27b",
                "messages": [
                    {"role": "system", "content": "project rules [[x-kv-save: false]]"},
                    {"role": "user", "content": "request"},
                ],
                "tools": [],
            }
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(body),
                    headers={"Content-Type": "application/json", "X-Session-Id": "session-tag"},
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
            finally:
                connection.close()
            self.assertEqual(FakeLlamaHandler.saved, [])
            self.assertNotIn("session-tag", self.proxy.session_states)
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy


if __name__ == "__main__":
    unittest.main()
