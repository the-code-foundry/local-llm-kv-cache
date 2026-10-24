#!/usr/bin/env python3
"""Session-aware proxy adding disk-backed llama.cpp slot snapshots.

Pi/Zed requests get one serialized operation, an optional slot restore, explicit
prompt caching, and a post-response snapshot. Requests without an explicit
session header use a stable-prefix-derived local affinity; media requests skip
disk snapshots because their prompt prefix is not reusable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cache_core import (
    PrefixManifest,
    best_prefix_manifest,
    build_prefix_payload,
    cache_filename,
    cache_key,
    manifest_filename,
    normalize_tokens,
    with_slot_cache,
)


LOGGER = logging.getLogger("local-llm-kv-cache")
DEFAULT_CACHE_DIR = str(Path.home() / ".llama-slot-cache")
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
MAX_METADATA_BYTES = 1024 * 1024
MAX_PENDING_PREFIX_SEEDS = 16


def _session_ref(session_id: str) -> str:
    """Return a pseudonymous, log-safe reference for a client session."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def _log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    payload = {"event": event, **{key: value for key, value in fields.items() if value is not None}}
    LOGGER.log(level, "%s", json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class KvControls:
    """Per-request cache behaviour, resolved header > body > inline tag > env."""

    save: bool = True
    restore: bool = True
    save_growth: int = 0
    save_interval: int = 0
    restore_tokens: int = 0


_KV_CONTROL_KEYS = ("save", "restore", "save-growth", "save-interval", "restore-tokens")
_KV_BOOL_KEYS = {"save", "restore"}


def _parse_kv_value(key: str, raw: str) -> bool | int | None:
    """Parse one control value; None (with a log) for anything invalid."""
    value = raw.strip().lower()
    if key in _KV_BOOL_KEYS:
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    else:
        try:
            parsed = int(value)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    _log_event(
        "kv_control_invalid",
        level=logging.WARNING,
        key=key,
        value=raw.strip(),
    )
    return None


def _kv_tag_pattern(inline_prefix: str) -> re.Pattern[str]:
    return re.compile(
        r"\[\[\s*" + re.escape(inline_prefix) + r"x-kv-([a-z][a-z-]*)\s*:\s*([^\[\]]*?)\s*\]\]",
        re.IGNORECASE,
    )


def _strip_kv_tags_from_text(text: str, pattern: re.Pattern[str], found: dict[str, bool | int]) -> str:
    def _consume(match: re.Match[str]) -> str:
        key = match.group(1).lower()
        if key in _KV_CONTROL_KEYS:
            value = _parse_kv_value(key, match.group(2))
            if value is not None:
                found[key] = value
        else:
            _log_event("kv_tag_unknown", level=logging.WARNING, key=key)
        return ""

    return pattern.sub(_consume, text)


def _extract_kv_tags(body: dict[str, Any], inline_prefix: str) -> dict[str, bool | int]:
    """Collect and strip ``[[x-kv-*: ...]]`` tags from the messages a request
    owner controls: every system message and the first non-system message.
    Tags elsewhere are ordinary content and are forwarded untouched. ``body``
    is mutated in place and must already be a private copy."""
    pattern = _kv_tag_pattern(inline_prefix)
    found: dict[str, bool | int] = {}
    saw_first_non_system = False
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        is_system = message.get("role") == "system"
        if not is_system:
            if saw_first_non_system:
                break
            saw_first_non_system = True
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _strip_kv_tags_from_text(content, pattern, found)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = _strip_kv_tags_from_text(part["text"], pattern, found)
    return found


def _extract_kv_body_fields(body: dict[str, Any]) -> dict[str, bool | int]:
    """Collect and strip ``kv_save``-style fields from the body and its
    ``extra_body``. ``body`` is mutated in place and must be a private copy."""
    found: dict[str, bool | int] = {}
    containers = [body]
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        containers.append(extra_body)
    for container in containers:
        for key in _KV_CONTROL_KEYS:
            field_name = "kv_" + key.replace("-", "_")
            if field_name not in container:
                continue
            raw = container.pop(field_name)
            value = _parse_kv_value(key, str(raw))
            if value is not None and key not in found:
                found[key] = value
    if isinstance(extra_body, dict) and not extra_body:
        body.pop("extra_body", None)
    return found


def _extract_kv_headers(headers: Any) -> dict[str, bool | int]:
    found: dict[str, bool | int] = {}
    for key in _KV_CONTROL_KEYS:
        header_name = "X-KV-" + "-".join(part.capitalize() for part in key.split("-"))
        raw = headers.get(header_name)
        if raw is None:
            continue
        value = _parse_kv_value(key, raw)
        if value is not None:
            found[key] = value
    return found


def _resolve_kv_controls(
    defaults: KvControls,
    *channels: dict[str, bool | int],
) -> KvControls:
    """Resolve per-request controls; earlier channels win over later ones."""
    resolved: dict[str, bool | int] = {}
    for channel in reversed(channels):
        resolved.update(channel)
    return KvControls(
        save=bool(resolved.get("save", defaults.save)),
        restore=bool(resolved.get("restore", defaults.restore)),
        save_growth=int(resolved.get("save-growth", defaults.save_growth)),
        save_interval=int(resolved.get("save-interval", defaults.save_interval)),
        restore_tokens=int(resolved.get("restore-tokens", defaults.restore_tokens)),
    )


def _is_slot_admin_path(path: str) -> bool:
    normalized = path.partition("?")[0].rstrip("/")
    return normalized == "/slots" or normalized.startswith("/slots/")


_ROUTE_ALIASES = {
    ("GET", "/v1/props"): "/props",
    ("HEAD", "/v1/props"): "/props",
    ("POST", "/v1/tokenize"): "/tokenize",
    ("POST", "/v1/detokenize"): "/detokenize",
    ("POST", "/v1/apply-template"): "/apply-template",
}
_NON_EVICTING_POST_PATHS = frozenset(
    {
        "/tokenize",
        "/detokenize",
        "/apply-template",
        "/v1/responses/input_tokens",
        "/v1/chat/completions/input_tokens",
        "/v1/messages/count_tokens",
        "/v1/chat/completions/control",
    }
)
_UNMANAGED_INFERENCE_POST_PATHS = frozenset(
    {
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
    }
)
_ADMIN_PATHS = frozenset({"/tools", "/lora-adapters"})


def _upstream_path(method: str, path: str) -> str:
    raw_path, separator, query = path.partition("?")
    alias = _ROUTE_ALIASES.get((method, raw_path.rstrip("/")))
    if alias is None:
        return path
    return alias + (f"?{query}" if separator else "")


def _normalized_path(path: str) -> str:
    return path.partition("?")[0].rstrip("/")


def _is_non_evicting_post_path(path: str) -> bool:
    normalized = _normalized_path(_upstream_path("POST", path))
    return normalized in _NON_EVICTING_POST_PATHS


def _is_known_post_path(path: str) -> bool:
    normalized = _normalized_path(_upstream_path("POST", path))
    return (
        normalized == "/v1/chat/completions"
        or normalized in _NON_EVICTING_POST_PATHS
        or normalized in _UNMANAGED_INFERENCE_POST_PATHS
    )


def _is_blocked_path(method: str, path: str) -> bool:
    normalized = _normalized_path(path)
    if _is_slot_admin_path(path) or normalized in _ADMIN_PATHS:
        return True
    if method in {"POST", "DELETE"} and (
        normalized == "/props" or normalized == "/models" or normalized.startswith("/models/")
    ):
        return True
    return method == "DELETE"


@dataclass(frozen=True)
class CompletionMetadata:
    finish_reason: str | None = None
    cached_tokens: int | None = None


@dataclass(frozen=True)
class ForwardResult:
    status: int
    metadata: CompletionMetadata = field(default_factory=CompletionMetadata)


def _metadata_from_object(value: Any) -> CompletionMetadata:
    if not isinstance(value, dict):
        return CompletionMetadata()
    finish_reason = None
    choices = value.get("choices")
    if not isinstance(choices, list):
        choices = []
    for choice in choices:
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
            finish_reason = choice["finish_reason"]
    cached_tokens = None
    usage = value.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            cached_tokens = details["cached_tokens"]
    return CompletionMetadata(finish_reason, cached_tokens)


def _completion_metadata(chunks: list[bytes]) -> CompletionMetadata:
    raw = b"".join(chunks)
    try:
        return _metadata_from_object(json.loads(raw or b"{}"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    result = CompletionMetadata()
    for line in raw.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            current = _metadata_from_object(json.loads(payload))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        result = CompletionMetadata(
            current.finish_reason or result.finish_reason,
            current.cached_tokens if current.cached_tokens is not None else result.cached_tokens,
        )
    return result


@dataclass(frozen=True)
class SnapshotPlan:
    session_id: str
    slot_id: int | None
    prefix_key: str
    session_file: Path
    prefix_file: Path
    prefix_payload: dict[str, Any]
    prefix_was_present: bool
    candidate_slot_id: int | None = None
    slot_task_ids: dict[int, int] = field(default_factory=dict)
    shared_prefix_candidate_tokens: int | None = None
    shared_prefix_verified_lcp: int | None = None
    stable_prefix_seed_tokens: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SlotState:
    slot_id: int
    prefix_key: str
    n_tokens: int
    session_file: Path | None = None
    dirty: bool = False
    # Token count of the session snapshot actually on disk; growth for
    # kv-save-growth is measured against this, not against the hot slot.
    saved_n_tokens: int | None = None
    # Monotonic time of the last persisted save; the kv-save-interval gate
    # is measured against this. None (e.g. after a restart) means no gate.
    saved_at: float | None = None


@dataclass(frozen=True)
class PendingPrefixSeed:
    prefix_payload: dict[str, Any]
    prefix_file: Path
    excluded_slot_id: int | None
    stable_tokens: tuple[int, ...] | None


class _NoIdleSlot(RuntimeError):
    """Signal that all llama slots are reserved by foreground requests."""


class LlamaCacheProxy:
    def __init__(
        self,
        upstream: str = "http://127.0.0.1:8080",
        cache_dir: str | None = None,
        max_cache_gib: float = 12.0,
        wait_seconds: float = 120.0,
        enable_prefix_seeding: bool = True,
        prefix_seed_delay_seconds: float = 2.0,
        save_policy: str = "all",
        require_session_id: bool = False,
        shared_prefix_scope: str | None = None,
        minimum_shared_prefix_tokens: int = 128,
        prefix_seed_timeout_seconds: float = 600.0,
        enable_shared_prefixes: bool = True,
        anonymous_affinity: str = "project",
        kv_controls: KvControls | None = None,
        inline_prefix: str = "",
    ) -> None:
        parsed = urlsplit(upstream)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("upstream must be an http URL")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or 80
        self.upstream_prefix = parsed.path.rstrip("/")
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.cache_dir.chmod(0o700)
        self.max_cache_bytes = int(max_cache_gib * 1024**3)
        self.wait_seconds = wait_seconds
        self.enable_prefix_seeding = enable_prefix_seeding
        self.enable_shared_prefixes = enable_shared_prefixes
        if anonymous_affinity not in {"project", "conversation-start"}:
            raise ValueError("anonymous_affinity must be 'project' or 'conversation-start'")
        self.anonymous_affinity = anonymous_affinity
        self.kv_controls = kv_controls or KvControls()
        if (
            self.kv_controls.save_growth < 0
            or self.kv_controls.save_interval < 0
            or self.kv_controls.restore_tokens < 0
        ):
            raise ValueError("kv_controls thresholds must be non-negative")
        self.inline_prefix = inline_prefix
        self.prefix_seed_delay_seconds = prefix_seed_delay_seconds
        if (
            isinstance(prefix_seed_timeout_seconds, bool)
            or not isinstance(prefix_seed_timeout_seconds, (int, float))
            or not math.isfinite(prefix_seed_timeout_seconds)
            or prefix_seed_timeout_seconds <= 0
        ):
            raise ValueError("prefix_seed_timeout_seconds must be a finite positive number")
        self.prefix_seed_timeout_seconds = float(prefix_seed_timeout_seconds)
        if save_policy not in {"all", "terminal"}:
            raise ValueError("save_policy must be 'all' or 'terminal'")
        self.save_policy = save_policy
        self.require_session_id = require_session_id
        self.cache_namespace = os.environ.get("PI_LLAMA_CACHE_NAMESPACE", "default").strip() or "default"
        self.shared_prefix_scope = (
            shared_prefix_scope
            if shared_prefix_scope is not None
            else os.environ.get("PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE", "default")
        ).strip()
        if not self.shared_prefix_scope:
            raise ValueError("shared_prefix_scope must not be empty")
        if (
            isinstance(minimum_shared_prefix_tokens, bool)
            or not isinstance(minimum_shared_prefix_tokens, int)
            or minimum_shared_prefix_tokens < 1
        ):
            raise ValueError("minimum_shared_prefix_tokens must be a positive integer")
        self.minimum_shared_prefix_tokens = minimum_shared_prefix_tokens
        self.operation_lock = threading.Lock()
        self.session_states: dict[str, SlotState] = {}
        # PI_LLAMA_CACHE_ENGINE_HOLDS=N (0 = off): the engine itself keeps the last N
        # conversations it served in RAM and resumes them by prompt match (Strata's
        # conversation cache), and its slot metadata cannot prove which one it holds.
        # Then a session served since the proxy started is sent straight through, no
        # disk restore; the live one is saved before another takes the engine; a disk
        # restore (which replaces the live conversation unparked) forgets the live one.
        self.engine_holds = max(0, int(os.environ.get("PI_LLAMA_CACHE_ENGINE_HOLDS", "0") or 0))
        self.live_session: str | None = None
        self.active_slot_ids: set[int] = set()
        self.active_slot_snapshots: dict[int, Path] = {}
        self.seed_slot_ids: set[int] = set()
        self.seed_slot_snapshots: dict[int, Path] = {}
        self.seed_token_reservations: set[tuple[int, ...]] = set()
        self.session_locks: dict[str, tuple[threading.Lock, int]] = {}
        self.session_locks_guard = threading.Lock()
        self.prefix_seed_lock = threading.Lock()
        self.prefix_seeds_in_flight: set[Path] = set()
        self.pending_prefix_seeds: dict[Path, PendingPrefixSeed] = {}
        self.foreground_waiters = 0
        self.foreground_waiters_lock = threading.Lock()
        self._prune()

    def prepare(
        self,
        body: dict[str, Any],
        session_id: str,
        *,
        excluded_slot_ids: set[int] | None = None,
        wait_for_slot: bool = True,
        controls: KvControls | None = None,
    ) -> tuple[dict[str, Any], SnapshotPlan]:
        controls = controls or self.kv_controls
        prefix_key = cache_key(body)
        session_file = self.cache_dir / cache_filename(session_id, body, "session")
        prefix_file = self.cache_dir / cache_filename("prefix", body, "prefix")
        prefix_payload = build_prefix_payload(body)
        session_ref = _session_ref(session_id)
        if not controls.save:
            # A deferred (dirty) state extended by a no-save request would later
            # be flushed with that request's content included. Persist the
            # deferred state first so the no-save request stays disposable.
            state = self.session_states.get(session_id)
            if state is not None and state.dirty:
                self._flush_dirty_states("before_no_save_request")
        if self.engine_holds and self.live_session not in (None, session_id):
            self._flush_dirty_states("before_switch")
        hot_state = self._hot_state(self.session_states.get(session_id), prefix_key)
        restored_source = None
        candidate_slot_id = None
        slot_task_ids: dict[int, int] = {}
        shared_prefix_restored = False
        shared_prefix_candidate_tokens = None
        shared_prefix_verified_lcp = None
        stable_prefix_seed_tokens = None
        request_tokens: tuple[int, ...] = ()
        rejected_shared_source: Path | None = None
        if hot_state is not None:
            slot_id = hot_state.slot_id
            self._touch_snapshot(session_file)
            _log_event("cache_hit", layer="hot", session_ref=session_ref, slot_id=slot_id)
        else:
            hot_state = None
            # A native cold request may choose any idle slot. Persist every dirty
            # owner before allowing llama.cpp to overwrite one.
            self._flush_dirty_states("before_eviction", excluded_slot_ids)
            candidate_slot_id = self._wait_for_idle_slot(excluded_slot_ids, wait=wait_for_slot)
            if not controls.restore:
                _log_event("kv_restore_disabled", session_ref=session_ref)
            if controls.restore and session_file.exists():
                previous_state = self.session_states.get(session_id)
                known_snapshot_tokens = (
                    previous_state.saved_n_tokens
                    if previous_state is not None and previous_state.session_file == session_file
                    else None
                )
                if (
                    controls.restore_tokens > 0
                    and known_snapshot_tokens is not None
                    and known_snapshot_tokens < controls.restore_tokens
                ):
                    _log_event(
                        "kv_restore_below_threshold",
                        session_ref=session_ref,
                        snapshot_tokens=known_snapshot_tokens,
                        threshold=controls.restore_tokens,
                    )
                else:
                    self._forget_slot(candidate_slot_id)
                    restored_source = self._restore_first_available(
                        candidate_slot_id, (session_file,), session_ref=session_ref
                    )
            if restored_source is None and controls.restore and self.enable_shared_prefixes:
                try:
                    request_tokens = self._render_tokens(body)
                    shared = self._best_shared_prefix(request_tokens)
                except (RuntimeError, TimeoutError, OSError, TypeError, ValueError) as error:
                    shared = None
                    _log_event(
                        "shared_prefix_discovery_failed",
                        level=logging.WARNING,
                        session_ref=session_ref,
                        error_type=type(error).__name__,
                    )
                if (
                    shared is not None
                    and controls.restore_tokens > 0
                    and shared[1] < controls.restore_tokens
                ):
                    _log_event(
                        "kv_restore_below_threshold",
                        session_ref=session_ref,
                        snapshot_tokens=shared[1],
                        threshold=controls.restore_tokens,
                    )
                    shared = None
                if shared is not None:
                    shared_source, candidate_tokens, verified_lcp = shared
                    if verified_lcp < candidate_tokens:
                        stable_prefix_seed_tokens = request_tokens[:verified_lcp]
                        rejected_shared_source = shared_source
                        _log_event(
                            "shared_prefix_overlap_discovered",
                            session_ref=session_ref,
                            candidate_tokens=candidate_tokens,
                            verified_lcp=verified_lcp,
                        )
                    else:
                        self._forget_slot(candidate_slot_id)
                        restored_source = self._restore_first_available(
                            candidate_slot_id,
                            (shared_source,),
                            session_ref=session_ref,
                            expected_tokens=candidate_tokens,
                            remove_on_count_mismatch=True,
                        )
                        if restored_source is not None:
                            shared_prefix_restored = True
                            shared_prefix_candidate_tokens = candidate_tokens
                            shared_prefix_verified_lcp = verified_lcp
                            _log_event(
                                "shared_prefix_candidate_restored",
                                session_ref=session_ref,
                                slot_id=candidate_slot_id,
                                candidate_tokens=candidate_tokens,
                                verified_lcp=verified_lcp,
                            )
                        else:
                            rejected_shared_source = shared_source
            if (
                restored_source is None
                and controls.restore
                and self.enable_shared_prefixes
                and rejected_shared_source is None
                and self.shared_prefix_scope == "default"
                and prefix_file.exists()
                and not prefix_file.with_name(manifest_filename(prefix_file.name)).exists()
            ):
                self._forget_slot(candidate_slot_id)
                restored_source = self._restore_first_available(
                    candidate_slot_id, (prefix_file,), session_ref=session_ref
                )
            if restored_source is None:
                _log_event("cache_miss", session_ref=session_ref, slot_id=candidate_slot_id)
            elif restored_source in (session_file, prefix_file):
                _log_event(
                    "cache_hit",
                    layer="session" if restored_source == session_file else "prefix",
                    session_ref=session_ref,
                    slot_id=candidate_slot_id,
                )
            slot_id = candidate_slot_id if restored_source is not None else None
            if slot_id is None:
                slot_task_ids = self._slot_task_ids()
        plan = SnapshotPlan(
            session_id=session_id,
            slot_id=slot_id,
            prefix_key=prefix_key,
            session_file=session_file,
            prefix_file=prefix_file,
            prefix_payload=prefix_payload,
            # Legacy exact-prefix snapshots deliberately remain "not present":
            # completion then schedules migration to a token-proven manifest.
            prefix_was_present=shared_prefix_restored,
            candidate_slot_id=candidate_slot_id,
            slot_task_ids=slot_task_ids,
            shared_prefix_candidate_tokens=shared_prefix_candidate_tokens,
            shared_prefix_verified_lcp=shared_prefix_verified_lcp,
            stable_prefix_seed_tokens=stable_prefix_seed_tokens,
        )
        return with_slot_cache(body, slot_id), plan

    def _render_tokens(self, body: dict[str, Any]) -> tuple[int, ...]:
        rendered = self._json_request("POST", "/apply-template", copy.deepcopy(body))
        prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("apply-template returned no prompt")
        tokenized = self._json_request(
            "POST",
            "/tokenize",
            {"content": prompt, "add_special": False, "parse_special": True},
        )
        if not isinstance(tokenized, dict):
            raise TypeError("tokenize returned an invalid response")
        return normalize_tokens(tokenized.get("tokens"))

    def _shared_prefix_candidates(self) -> list[tuple[PrefixManifest, Path]]:
        candidates: list[tuple[PrefixManifest, Path]] = []
        for path in self.cache_dir.glob("local-llm-prefix-*.bin.manifest.json"):
            try:
                if path.stat().st_size > MAX_METADATA_BYTES:
                    continue
                manifest = PrefixManifest.from_json(path.read_bytes())
                snapshot = self.cache_dir / manifest.snapshot
                if path.name != f"{manifest.snapshot}.manifest.json" or not snapshot.is_file():
                    continue
                candidates.append((manifest, snapshot))
            except (OSError, ValueError):
                continue
        return candidates

    def _best_shared_prefix(self, request_tokens: tuple[int, ...]) -> tuple[Path, int, int] | None:
        candidates = self._shared_prefix_candidates()
        selected = best_prefix_manifest(
            [manifest for manifest, _snapshot in candidates],
            request_tokens,
            self.cache_namespace,
            self.shared_prefix_scope,
            self.minimum_shared_prefix_tokens,
        )
        if selected is None:
            return None
        manifest, lcp = selected
        return self.cache_dir / manifest.snapshot, len(manifest.tokens), lcp

    def _published_shared_prefix(self, tokens: tuple[int, ...]) -> Path | None:
        for manifest, snapshot in self._shared_prefix_candidates():
            if (
                manifest.namespace == self.cache_namespace
                and manifest.scope == self.shared_prefix_scope
                and manifest.tokens == tokens
            ):
                return snapshot
        return None

    @staticmethod
    def _touch_snapshot(source: Path) -> int | None:
        if not source.exists():
            return None
        try:
            source.touch()
            return source.stat().st_size
        except OSError as error:
            _log_event(
                "snapshot_touch_failed",
                level=logging.WARNING,
                filename=source.name,
                error_type=type(error).__name__,
            )
            return None

    def prepare_uncached(self, reason: str = "before_uncached") -> None:
        """Protect dirty slots before a request that bypasses snapshot tracking."""
        self._flush_dirty_states(reason)
        invalidated = len(self.session_states)
        self.session_states.clear()
        _log_event("slot_ownership_reset", reason=reason, invalidated_sessions=invalidated)

    def _restore_first_available(
        self,
        slot_id: int,
        sources: tuple[Path, ...],
        session_ref: str | None = None,
        expected_tokens: int | None = None,
        remove_on_count_mismatch: bool = False,
    ) -> Path | None:
        for source in sources:
            if not source.exists():
                continue
            started = time.monotonic()
            try:
                n_restored = self._restore(slot_id, source)
            except (RuntimeError, TimeoutError, OSError) as error:
                _log_event(
                    "snapshot_restore_failed",
                    level=logging.WARNING,
                    session_ref=session_ref,
                    filename=source.name,
                    slot_id=slot_id,
                    error_type=type(error).__name__,
                )
                continue
            if expected_tokens is not None and n_restored != expected_tokens:
                _log_event(
                    "snapshot_restore_token_count_mismatch",
                    level=logging.ERROR,
                    session_ref=session_ref,
                    filename=source.name,
                    slot_id=slot_id,
                    expected_tokens=expected_tokens,
                    restored_tokens=n_restored,
                )
                if remove_on_count_mismatch:
                    for stale_path in (
                        source,
                        self.cache_dir / manifest_filename(source.name),
                    ):
                        try:
                            stale_path.unlink(missing_ok=True)
                        except OSError as exc:
                            _log_event(
                                "snapshot_pair_cleanup_failed",
                                level=logging.WARNING,
                                filename=stale_path.name,
                                error_type=type(exc).__name__,
                            )
                continue
            snapshot_bytes = self._touch_snapshot(source)
            _log_event(
                "snapshot_restore",
                session_ref=session_ref,
                filename=source.name,
                slot_id=slot_id,
                tokens=n_restored,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                snapshot_bytes=snapshot_bytes,
            )
            return source
        return None

    def finish(
        self,
        plan: SnapshotPlan,
        status: int,
        finish_reason: str | None = None,
        cached_tokens: int | None = None,
        *,
        prune: bool = True,
        controls: KvControls | None = None,
    ) -> None:
        if status < 200 or status >= 300:
            return
        controls = controls or self.kv_controls
        slot_id = self._resolve_slot(plan)
        if plan.slot_id is None:
            self._forget_slot(slot_id)

        session_ref = _session_ref(plan.session_id)
        if not controls.save:
            # The request's state is disposable: no save, no deferral, and no
            # session-state update, so the next request restores the snapshot
            # already on disk instead of continuing from this slot.
            _log_event(
                "request_complete",
                session_ref=session_ref,
                slot_id=slot_id,
                finish_reason=finish_reason,
                cached_tokens=cached_tokens,
                snapshot_saved=False,
                kv_save="disabled",
            )
            if not plan.prefix_was_present:
                self._schedule_prefix_seed(replace(plan, slot_id=slot_id))
            self._retry_pending_prefix_seeds()
            return
        previous = self.session_states.get(plan.session_id)
        previous_matches = (
            previous is not None
            and previous.prefix_key == plan.prefix_key
            and previous.session_file == plan.session_file
        )
        previous_saved_tokens = previous.saved_n_tokens if previous_matches else None
        previous_saved_at = previous.saved_at if previous_matches else None
        defer = self.save_policy == "terminal" and finish_reason == "tool_calls"
        did_save = False
        if defer:
            n_tokens = self._slot_token_count(slot_id)
            if n_tokens > 0:
                state = SlotState(
                    slot_id,
                    plan.prefix_key,
                    n_tokens,
                    plan.session_file,
                    True,
                    saved_n_tokens=previous_saved_tokens,
                    saved_at=previous_saved_at,
                )
                _log_event(
                    "snapshot_deferred",
                    session_ref=session_ref,
                    slot_id=slot_id,
                    tokens=n_tokens,
                    finish_reason=finish_reason,
                    cached_tokens=cached_tokens,
                )
            else:
                # Unknown slot metadata must fail safe: persist instead of risking
                # eviction of an uncheckpointed conversation.
                n_saved = self._save(
                    slot_id,
                    plan.session_file,
                    reason="metadata_fallback",
                    session_ref=session_ref,
                )
                state = SlotState(
                    slot_id,
                    plan.prefix_key,
                    n_saved,
                    plan.session_file,
                    False,
                    saved_n_tokens=n_saved,
                    saved_at=time.monotonic(),
                )
                did_save = True
        elif self._should_skip_save(
            previous_saved_tokens, previous_saved_at, plan, slot_id, controls.save_growth, controls.save_interval
        ):
            # The slot grew too little, or too recently, to be worth rewriting a
            # multi-GB file. The state stays dirty so eviction and shutdown still
            # persist it, and growth and age keep being measured against the
            # snapshot on disk.
            n_tokens = self._slot_token_count(slot_id)
            state = SlotState(
                slot_id,
                plan.prefix_key,
                n_tokens,
                plan.session_file,
                True,
                saved_n_tokens=previous_saved_tokens,
                saved_at=previous_saved_at,
            )
            _log_event(
                "snapshot_save_skipped",
                session_ref=session_ref,
                slot_id=slot_id,
                tokens=n_tokens,
                snapshot_tokens=previous_saved_tokens,
                threshold=controls.save_growth,
                interval=controls.save_interval or None,
                age_seconds=(
                    round(time.monotonic() - previous_saved_at, 3) if previous_saved_at is not None else None
                ),
            )
        else:
            n_saved = self._save(
                slot_id,
                plan.session_file,
                reason="response_complete",
                session_ref=session_ref,
            )
            state = SlotState(
                slot_id,
                plan.prefix_key,
                n_saved,
                plan.session_file,
                False,
                saved_n_tokens=n_saved,
                saved_at=time.monotonic(),
            )
            did_save = True
        self.session_states[plan.session_id] = state
        if self.engine_holds:
            self._hold(plan.session_id)
        _log_event(
            "request_complete",
            session_ref=session_ref,
            slot_id=slot_id,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            snapshot_saved=did_save,
        )
        if plan.shared_prefix_candidate_tokens is not None:
            outcome = (
                "shared_prefix_effective_hit"
                if cached_tokens is not None and cached_tokens > 0
                else "shared_prefix_rejected_by_llama"
                if cached_tokens == 0
                else "shared_prefix_effectiveness_unknown"
            )
            _log_event(
                outcome,
                session_ref=session_ref,
                slot_id=slot_id,
                candidate_tokens=plan.shared_prefix_candidate_tokens,
                verified_lcp=plan.shared_prefix_verified_lcp,
                cached_tokens=cached_tokens,
            )
        if not plan.prefix_was_present or (
            plan.shared_prefix_candidate_tokens is not None and cached_tokens == 0
        ):
            self._schedule_prefix_seed(replace(plan, slot_id=slot_id))
        self._retry_pending_prefix_seeds()
        if did_save and prune:
            self._prune()

    def _should_skip_save(
        self,
        previous_saved_tokens: int | None,
        previous_saved_at: float | None,
        plan: SnapshotPlan,
        slot_id: int,
        save_growth: int,
        save_interval: int,
    ) -> bool:
        """A save is skipped when any enabled gate says it is too early: the
        growth gate (the slot grew fewer than ``save_growth`` tokens over the
        snapshot on disk) or the interval gate (the last persisted save was
        less than ``save_interval`` seconds ago). A gate without a baseline
        (no snapshot, unknown slot metadata, no recorded save time) never
        skips: failing safe means saving."""
        if not plan.session_file.exists():
            return False
        if save_growth > 0 and previous_saved_tokens is not None:
            current = self._slot_token_count(slot_id)
            # Unknown slot metadata must fail safe: save.
            if current > 0 and current - previous_saved_tokens < save_growth:
                return True
        if save_interval > 0 and previous_saved_at is not None:
            if time.monotonic() - previous_saved_at < save_interval:
                return True
        return False

    def _slot_token_count(self, slot_id: int) -> int:
        try:
            slots = self._slots()
        except (RuntimeError, TimeoutError, OSError, TypeError, ValueError) as error:
            _log_event(
                "slot_metadata_unavailable",
                level=logging.WARNING,
                slot_id=slot_id,
                error_type=type(error).__name__,
            )
            return 0
        for slot in slots:
            if int(slot.get("id", -1)) == slot_id and not slot.get("is_processing"):
                return int(slot.get("n_prompt_tokens") or 0)
        return 0

    def _flush_dirty_states(self, reason: str, excluded_slot_ids: set[int] | None = None) -> None:
        excluded = excluded_slot_ids or set()
        for session_id, state in list(self.session_states.items()):
            if not state.dirty or state.session_file is None or state.slot_id in excluded:
                continue
            session_ref = _session_ref(session_id)
            n_saved = self._save(
                state.slot_id,
                state.session_file,
                reason=reason,
                session_ref=session_ref,
            )
            self.session_states[session_id] = replace(
                state, n_tokens=n_saved, dirty=False, saved_n_tokens=n_saved, saved_at=time.monotonic()
            )
            _log_event(
                "snapshot_flush",
                session_ref=session_ref,
                slot_id=state.slot_id,
                filename=state.session_file.name,
                reason=reason,
                tokens=n_saved,
            )

    def flush_dirty(self, reason: str = "shutdown") -> None:
        with self.foreground_operation():
            self._flush_dirty_states(reason)
            self._prune()

    def _hot_state(self, state: SlotState | None, prefix_key: str) -> SlotState | None:
        if state is None or state.prefix_key != prefix_key:
            return None
        if self.engine_holds:
            return state
        for slot in self._slots():
            if int(slot.get("id", -1)) != state.slot_id:
                continue
            if slot.get("is_processing"):
                return None
            if int(slot.get("n_prompt_tokens") or 0) != state.n_tokens:
                return None
            return state
        return None

    def _forget_slot(self, slot_id: int) -> None:
        if self.engine_holds:
            return  # one engine, many held conversations: see _restore and _hold
        self.session_states = {
            key: state for key, state in self.session_states.items() if state.slot_id != slot_id
        }

    @staticmethod
    def _prefix_seed_payload(body: dict[str, Any]) -> dict[str, Any]:
        payload = build_prefix_payload(body)
        payload.update(
            {
                "add_generation_prompt": False,
                "cache_prompt": False,
                "n_predict": 0,
                "stream": False,
            }
        )
        return payload

    def _schedule_prefix_seed(self, plan: SnapshotPlan) -> None:
        if not self.enable_shared_prefixes or not self.enable_prefix_seeding:
            return
        # llama.cpp chat templates require at least one message. A tools-only
        # stable prefix cannot be rendered safely and must remain a cold miss.
        if not plan.prefix_payload.get("messages"):
            return
        pending = PendingPrefixSeed(
            plan.prefix_payload,
            plan.prefix_file,
            plan.slot_id,
            plan.stable_prefix_seed_tokens,
        )
        with self.prefix_seed_lock:
            existing = self.pending_prefix_seeds.get(plan.prefix_file)
            if existing is None and len(self.pending_prefix_seeds) >= MAX_PENDING_PREFIX_SEEDS:
                _log_event(
                    "prefix_seed_queue_full",
                    level=logging.WARNING,
                    filename=plan.prefix_file.name,
                    limit=MAX_PENDING_PREFIX_SEEDS,
                )
                return
            if (
                existing is not None
                and existing.stable_tokens is not None
                and (
                    pending.stable_tokens is None
                    or len(existing.stable_tokens) >= len(pending.stable_tokens)
                )
            ):
                pending = replace(pending, stable_tokens=existing.stable_tokens)
            self.pending_prefix_seeds[plan.prefix_file] = pending
        self._start_pending_prefix_seed(plan.prefix_file)

    def _retry_pending_prefix_seeds(self) -> None:
        with self.prefix_seed_lock:
            pending_files = tuple(self.pending_prefix_seeds)
        for prefix_file in pending_files:
            self._start_pending_prefix_seed(prefix_file)

    def _start_pending_prefix_seed(self, prefix_file: Path) -> None:
        with self.prefix_seed_lock:
            pending = self.pending_prefix_seeds.get(prefix_file)
            if pending is None or prefix_file in self.prefix_seeds_in_flight:
                return
            self.prefix_seeds_in_flight.add(prefix_file)
        threading.Thread(
            target=self._seed_prefix,
            args=(
                pending.prefix_payload,
                pending.prefix_file,
                pending.excluded_slot_id,
                pending.stable_tokens,
            ),
            name="local-llm-kv-prefix-seed",
            daemon=True,
        ).start()

    def _seed_prefix(
        self,
        prefix_payload: dict[str, Any],
        prefix_file: Path,
        excluded_slot_id: int | None,
        stable_tokens: tuple[int, ...] | None = None,
    ) -> None:
        slot_id: int | None = None
        tokens: tuple[int, ...] | None = None
        try:
            time.sleep(self.prefix_seed_delay_seconds)
            if self._has_foreground_waiters():
                LOGGER.info("prefix seed skipped while foreground request is waiting: %s", prefix_file.name)
                return
            if not self.operation_lock.acquire(blocking=False):
                LOGGER.info("prefix seed skipped while proxy is busy: %s", prefix_file.name)
                return
            try:
                if not self._seed_spare_slots(excluded_slot_id):
                    LOGGER.info("prefix seed skipped: no safe idle slot: %s", prefix_file.name)
                    return
            finally:
                self.operation_lock.release()
            if stable_tokens is None:
                request = self._prefix_seed_payload(prefix_payload)
                tokens = self._render_tokens(request)
            else:
                tokens = normalize_tokens(stable_tokens)
            if not self.operation_lock.acquire(blocking=False):
                LOGGER.info("prefix seed skipped while proxy is busy: %s", prefix_file.name)
                return
            try:
                published = self._published_shared_prefix(tokens)
                if published is not None:
                    _log_event(
                        "prefix_seed_duplicate_skipped",
                        filename=published.name,
                        tokens=len(tokens),
                    )
                    self._discard_pending_prefix_seed(prefix_file, stable_tokens)
                    return
                if tokens in self.seed_token_reservations:
                    LOGGER.info("prefix seed skipped: equivalent seed is already reserved: %s", prefix_file.name)
                    return
                if self._has_foreground_waiters():
                    LOGGER.info("prefix seed skipped after discovery: foreground request is waiting: %s", prefix_file.name)
                    return
                spare_slots = self._seed_spare_slots(excluded_slot_id)
                if not spare_slots:
                    LOGGER.info("prefix seed skipped: no safe idle slot: %s", prefix_file.name)
                    return
                slot_id = int(
                    min(spare_slots, key=lambda slot: int(slot.get("n_prompt_tokens") or 0)).get("id", 0)
                )
                seed_file = self._prefix_seed_target(prefix_file)
                self.seed_slot_ids.add(slot_id)
                self.seed_slot_snapshots[slot_id] = seed_file
                self.seed_token_reservations.add(tokens)
            finally:
                self.operation_lock.release()
            seed_request = {
                "prompt": list(tokens),
                "cache_prompt": False,
                "n_predict": 0,
                "stream": False,
                "id_slot": slot_id,
            }
            self._json_request(
                "POST",
                "/completion",
                seed_request,
                timeout=self.prefix_seed_timeout_seconds,
            )
            manifest_path = self.cache_dir / manifest_filename(seed_file.name)
            n_saved = self._save(slot_id, seed_file)
            if n_saved != len(tokens):
                seed_file.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
                _log_event(
                    "prefix_seed_token_count_mismatch",
                    level=logging.ERROR,
                    filename=seed_file.name,
                    slot_id=slot_id,
                    expected_tokens=len(tokens),
                    saved_tokens=n_saved,
                )
                return
            try:
                self._write_manifest(seed_file, tokens)
            except Exception:
                seed_file.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
                raise
            self._discard_pending_prefix_seed(prefix_file, stable_tokens)
            LOGGER.info("seeded stable prefix -> %s", prefix_file.name)
        except Exception:
            LOGGER.exception("prefix seed failed: %s", prefix_file.name)
        finally:
            try:
                if slot_id is not None:
                    with self.operation_lock:
                        self.seed_slot_ids.discard(slot_id)
                        self.seed_slot_snapshots.pop(slot_id, None)
                        if tokens is not None:
                            self.seed_token_reservations.discard(tokens)
                        self._forget_slot(slot_id)
                        self._prune()
            finally:
                with self.prefix_seed_lock:
                    self.prefix_seeds_in_flight.discard(prefix_file)

    def _seed_spare_slots(self, excluded_slot_id: int | None) -> list[dict[str, Any]]:
        """Return idle slots that no foreground request, owner, or seed has reserved."""
        owned_slot_ids = {state.slot_id for state in self.session_states.values()}
        reserved_slot_ids = self.active_slot_ids | self.seed_slot_ids
        return [
            slot
            for slot in self._idle_slots()
            if int(slot.get("id", -1)) != excluded_slot_id
            and int(slot.get("id", -1)) not in owned_slot_ids
            and int(slot.get("id", -1)) not in reserved_slot_ids
        ]

    def _discard_pending_prefix_seed(
        self,
        prefix_file: Path,
        stable_tokens: tuple[int, ...] | None,
    ) -> None:
        with self.prefix_seed_lock:
            pending = self.pending_prefix_seeds.get(prefix_file)
            if pending is not None and pending.stable_tokens == stable_tokens:
                self.pending_prefix_seeds.pop(prefix_file, None)

    def _prefix_seed_target(self, prefix_file: Path) -> Path:
        """Keep a published pair immutable while a replacement pair is built."""
        manifest_path = self.cache_dir / manifest_filename(prefix_file.name)
        try:
            manifest = PrefixManifest.from_json(manifest_path.read_bytes())
            valid_pair = prefix_file.is_file() and manifest.snapshot == prefix_file.name
        except (OSError, ValueError):
            valid_pair = False
        if not valid_pair:
            return prefix_file
        generation = os.urandom(8).hex()
        return prefix_file.with_name(f"{prefix_file.stem}-{generation}{prefix_file.suffix}")

    def _write_manifest(self, snapshot: Path, tokens: tuple[int, ...]) -> Path:
        if not snapshot.is_file():
            raise RuntimeError("cannot publish a manifest without its snapshot")
        manifest = PrefixManifest(
            self.cache_namespace,
            self.shared_prefix_scope,
            snapshot.name,
            tokens,
        )
        target = self.cache_dir / manifest_filename(snapshot.name)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(manifest.to_json())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
            target.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def _has_foreground_waiters(self) -> bool:
        with self.foreground_waiters_lock:
            return self.foreground_waiters > 0

    def begin_request(
        self,
        body: dict[str, Any],
        session_id: str,
        controls: KvControls | None = None,
    ) -> tuple[dict[str, Any], SnapshotPlan]:
        """Atomically prepare and reserve one slot for a managed request."""
        deadline = time.monotonic() + self.wait_seconds
        while True:
            with self.operation_lock:
                try:
                    request, plan = self.prepare(
                        body,
                        session_id,
                        excluded_slot_ids=self.active_slot_ids | self.seed_slot_ids,
                        wait_for_slot=False,
                        controls=controls,
                    )
                except _NoIdleSlot:
                    pass
                else:
                    slot_id = plan.slot_id if plan.slot_id is not None else plan.candidate_slot_id
                    if slot_id is None:
                        raise RuntimeError("prepared request has no llama slot")
                    if slot_id in self.active_slot_ids or slot_id in self.seed_slot_ids:
                        raise RuntimeError(f"llama slot {slot_id} is already reserved")
                    if plan.slot_id is None:
                        self._forget_slot(slot_id)
                    self.active_slot_ids.add(slot_id)
                    reserved_plan = replace(plan, slot_id=slot_id)
                    self.active_slot_snapshots[slot_id] = reserved_plan.session_file
                    return with_slot_cache(request, slot_id), reserved_plan
            if time.monotonic() >= deadline:
                raise TimeoutError("no unreserved llama slot became available")
            time.sleep(0.05)

    def complete_request(
        self,
        plan: SnapshotPlan,
        status: int,
        finish_reason: str | None = None,
        cached_tokens: int | None = None,
        controls: KvControls | None = None,
    ) -> None:
        """Finalize a request before releasing its reserved slot."""
        with self.operation_lock:
            finalized = False
            try:
                self.finish(plan, status, finish_reason, cached_tokens, prune=False, controls=controls)
                finalized = 200 <= status < 300
            finally:
                if (
                    plan.slot_id is not None
                    and self.active_slot_snapshots.get(plan.slot_id) == plan.session_file
                ):
                    self.active_slot_ids.discard(plan.slot_id)
                    self.active_slot_snapshots.pop(plan.slot_id, None)
                if finalized:
                    self._prune()

    def cancel_request(self, plan: SnapshotPlan) -> None:
        """Release a reserved slot after forwarding failed."""
        with self.operation_lock:
            if (
                plan.slot_id is not None
                and self.active_slot_snapshots.get(plan.slot_id) == plan.session_file
            ):
                self.active_slot_ids.discard(plan.slot_id)
                self.active_slot_snapshots.pop(plan.slot_id, None)

    @contextmanager
    def session_operation(self, session_id: str):
        """Serialize one session while allowing different sessions to overlap."""
        with self.session_locks_guard:
            lock, users = self.session_locks.get(session_id, (threading.Lock(), 0))
            self.session_locks[session_id] = (lock, users + 1)
        with self.foreground_waiters_lock:
            self.foreground_waiters += 1
        try:
            lock.acquire()
        except BaseException:
            with self.foreground_waiters_lock:
                self.foreground_waiters -= 1
            with self.session_locks_guard:
                current_lock, users = self.session_locks[session_id]
                if users == 1:
                    self.session_locks.pop(session_id)
                else:
                    self.session_locks[session_id] = (current_lock, users - 1)
            raise
        try:
            yield
        finally:
            lock.release()
            with self.foreground_waiters_lock:
                self.foreground_waiters -= 1
            with self.session_locks_guard:
                current_lock, users = self.session_locks[session_id]
                if users == 1:
                    self.session_locks.pop(session_id)
                else:
                    self.session_locks[session_id] = (current_lock, users - 1)

    def _slot_task_ids(self) -> dict[int, int]:
        return {
            int(slot.get("id", -1)): int(slot.get("id_task") or -1)
            for slot in self._slots()
        }

    def _resolve_slot(self, plan: SnapshotPlan) -> int:
        if plan.slot_id is not None:
            return plan.slot_id
        try:
            slots = [slot for slot in self._slots() if not slot.get("is_processing")]
        except (RuntimeError, TimeoutError, OSError, TypeError, ValueError) as error:
            if plan.candidate_slot_id is None:
                raise
            _log_event(
                "slot_resolution_fallback",
                level=logging.WARNING,
                slot_id=plan.candidate_slot_id,
                error_type=type(error).__name__,
            )
            return plan.candidate_slot_id
        changed = [
            slot
            for slot in slots
            if int(slot.get("id_task") or -1) != plan.slot_task_ids.get(int(slot.get("id", -1)), -1)
        ]
        candidates = changed or slots
        if candidates:
            return int(max(candidates, key=lambda slot: int(slot.get("id_task") or -1)).get("id", 0))
        if plan.candidate_slot_id is not None:
            return plan.candidate_slot_id
        return self._wait_for_idle_slot()

    def _idle_slots(self, excluded_slot_ids: set[int] | None = None) -> list[dict[str, Any]]:
        excluded = excluded_slot_ids or set()
        return [
            slot
            for slot in self._slots()
            if not slot.get("is_processing") and int(slot.get("id", -1)) not in excluded
        ]

    @contextmanager
    def foreground_operation(self):
        with self.foreground_waiters_lock:
            self.foreground_waiters += 1
        try:
            deadline = time.monotonic() + self.wait_seconds
            while True:
                self.operation_lock.acquire()
                if not self.active_slot_ids and not self.seed_slot_ids:
                    break
                self.operation_lock.release()
                if time.monotonic() >= deadline:
                    raise TimeoutError("foreground requests did not drain")
                time.sleep(0.05)
        except BaseException:
            with self.foreground_waiters_lock:
                self.foreground_waiters -= 1
            raise
        with self.foreground_waiters_lock:
            self.foreground_waiters -= 1
        try:
            yield
        finally:
            self.operation_lock.release()

    def forward(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        path: str,
        body: bytes,
    ) -> ForwardResult:
        upstream = HTTPConnection(self.upstream_host, self.upstream_port, timeout=1200)
        headers = self._forward_headers(handler)
        is_head = method == "HEAD"
        upstream.request(method, f"{self.upstream_prefix}{path}", body=body, headers=headers)
        response = upstream.getresponse()
        setattr(handler, "_proxy_response_started", True)
        handler.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            normalized_key = key.lower()
            if normalized_key in HOP_BY_HOP_HEADERS and not (
                is_head and normalized_key == "content-length"
            ):
                continue
            handler.send_header(key, value)
        if not is_head:
            handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        metadata_buffer = bytearray()
        try:
            while True:
                # ``HTTPResponse.read(size)`` waits for ``size`` bytes or EOF and
                # therefore buffers small SSE events until generation completes.
                # ``read1`` returns data from a single underlying socket read, so
                # each available upstream fragment can be flushed immediately.
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                metadata_buffer.extend(chunk)
                if len(metadata_buffer) > MAX_METADATA_BYTES:
                    del metadata_buffer[:-MAX_METADATA_BYTES]
                if not is_head:
                    handler.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    handler.wfile.write(chunk)
                    handler.wfile.write(b"\r\n")
                    handler.wfile.flush()
            if not is_head:
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()
            return ForwardResult(response.status, _completion_metadata([bytes(metadata_buffer)]))
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.debug("client disconnected while proxying %s %s", method, path)
            return ForwardResult(response.status, _completion_metadata([bytes(metadata_buffer)]))
        finally:
            upstream.close()

    def _forward_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        headers = {"Content-Type": handler.headers.get("Content-Type", "application/json")}
        for name in (
            "Authorization",
            "Accept",
            "User-Agent",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
            "X-Api-Key",
            "Anthropic-Version",
            "Anthropic-Beta",
            "OpenAI-Organization",
            "OpenAI-Project",
            "Idempotency-Key",
        ):
            value = handler.headers.get(name)
            if value:
                headers[name] = value
        return headers

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        connection = HTTPConnection(self.upstream_host, self.upstream_port, timeout=timeout)
        try:
            connection.request(
                method,
                f"{self.upstream_prefix}{path}",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"llama {method} {path} returned {response.status}")
            return json.loads(raw or b"{}")
        finally:
            connection.close()

    def _slots(self) -> list[dict[str, Any]]:
        connection = HTTPConnection(self.upstream_host, self.upstream_port, timeout=10)
        try:
            connection.request("GET", f"{self.upstream_prefix}/slots")
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"llama GET /slots returned {response.status}")
            data = json.loads(response.read())
            return data if isinstance(data, list) else []
        finally:
            connection.close()

    def _wait_for_idle_slot(
        self,
        excluded_slot_ids: set[int] | None = None,
        *,
        wait: bool = True,
    ) -> int:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            slots = self._idle_slots(excluded_slot_ids)
            if slots:
                owned_slots = {state.slot_id for state in self.session_states.values()}
                unowned_slots = [slot for slot in slots if int(slot.get("id", -1)) not in owned_slots]
                candidates = unowned_slots or slots
                return min(candidates, key=lambda slot: int(slot.get("n_prompt_tokens") or 0)).get("id", 0)
            if not wait:
                raise _NoIdleSlot("no unreserved idle llama slot")
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        raise TimeoutError("no idle llama slot became available")

    def _restore(self, slot_id: int, filename: Path) -> int:
        result = self._json_request(
            "POST",
            f"/slots/{slot_id}?action=restore",
            {"filename": filename.name},
        )
        n_restored = int(result.get("n_restored") or 0)
        if n_restored <= 0:
            raise RuntimeError(f"llama restored no tokens from {filename.name}")
        if self.engine_holds and self.live_session is not None:
            # the restore replaced the live conversation without parking it
            self.session_states.pop(self.live_session, None)
            self.live_session = None
        return n_restored

    def _hold(self, session_id: str) -> None:
        """Engine-holds mode: session_id is the live conversation; keep the newest N."""
        self.live_session = session_id
        state = self.session_states.pop(session_id)
        self.session_states[session_id] = state  # newest last
        while len(self.session_states) > self.engine_holds:
            oldest = next(iter(self.session_states))
            dropped = self.session_states.pop(oldest)
            _log_event("engine_hold_dropped", session_ref=_session_ref(oldest), dirty=dropped.dirty)

    def _save(
        self,
        slot_id: int,
        target: Path,
        reason: str | None = None,
        session_ref: str | None = None,
    ) -> int:
        started = time.monotonic()
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            result = self._json_request(
                "POST",
                f"/slots/{slot_id}?action=save",
                {"filename": temporary.name},
            )
            if int(result.get("n_saved") or 0) <= 0 or not temporary.exists():
                raise RuntimeError(f"llama saved no tokens for slot {slot_id}")
            temporary.replace(target)
            target.chmod(0o600)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                _log_event(
                    "snapshot_temp_cleanup_failed",
                    level=logging.WARNING,
                    filename=temporary.name,
                    error_type=type(error).__name__,
                )
        n_saved = int(result["n_saved"])
        _log_event(
            "snapshot_save",
            filename=target.name,
            slot_id=slot_id,
            tokens=n_saved,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            snapshot_bytes=target.stat().st_size,
            reason=reason,
            session_ref=session_ref,
        )
        return n_saved

    def _prune(self) -> None:
        protected = set(self.active_slot_snapshots.values())
        protected.update(self.seed_slot_snapshots.values())
        protected_temporaries = {
            path.with_suffix(path.suffix + ".tmp") for path in self.seed_slot_snapshots.values()
        }
        protected.update(
            state.session_file
            for state in self.session_states.values()
            if state.dirty and state.session_file is not None
        )
        for manifest in self.cache_dir.glob("local-llm-prefix-*.bin.manifest.json"):
            snapshot = self.cache_dir / manifest.name.removesuffix(".manifest.json")
            if not snapshot.is_file():
                manifest.unlink(missing_ok=True)
        for temporary in self.cache_dir.glob("local-llm-*.tmp"):
            if temporary not in protected_temporaries:
                temporary.unlink(missing_ok=True)
        all_files = list(self.cache_dir.glob("local-llm-*.bin"))
        files = sorted(
            (path for path in all_files if path not in protected),
            key=lambda path: path.stat().st_mtime,
        )
        total = sum(path.stat().st_size for path in all_files)
        while total > self.max_cache_bytes and files:
            victim = files.pop(0)
            size = victim.stat().st_size
            victim.unlink(missing_ok=True)
            if victim.name.startswith("local-llm-prefix-"):
                (self.cache_dir / manifest_filename(victim.name)).unlink(missing_ok=True)
            total -= size
            _log_event(
                "snapshot_prune",
                filename=victim.name,
                snapshot_bytes=size,
                cache_bytes=total,
            )


def _session_id(handler: BaseHTTPRequestHandler) -> str | None:
    for header in (
        "X-Session-Affinity",
        "X-Session-Id",
        "X-Conversation-Id",
        "X-Pi-Session-Id",
        "X-OpenCode-Session",
        "X-Client-Request-Id",
    ):
        value = handler.headers.get(header)
        if value:
            return value.strip()
    return None


def _body_session_id(body: dict[str, Any]) -> str | None:
    containers = [body]
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        containers.append(extra_body)
    for container in containers:
        for affinity_field in ("session_id", "conversation_id", "prompt_cache_key"):
            value = container.get(affinity_field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _without_proxy_affinity_fields(body: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(body)
    for affinity_field in ("session_id", "conversation_id", "prompt_cache_key"):
        request.pop(affinity_field, None)
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict):
        for affinity_field in ("session_id", "conversation_id", "prompt_cache_key"):
            extra_body.pop(affinity_field, None)
        if not extra_body:
            request.pop("extra_body")
    return request


def _anonymous_session_id(body: dict[str, Any], affinity: str = "project") -> str:
    if affinity == "project":
        key = cache_key(body)
    elif affinity == "conversation-start":
        payload = build_prefix_payload(body)
        prefix_count = len(payload.get("messages") or [])
        for message in (body.get("messages") or [])[prefix_count:]:
            if isinstance(message, dict):
                payload["conversation_start"] = copy.deepcopy(message)
                break
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
    else:
        raise ValueError("anonymous affinity must be 'project' or 'conversation-start'")
    return f"anonymous-{key}"


def _has_media(body: dict[str, Any]) -> bool:
    if body.get("images"):
        return True
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(item, dict) and item.get("type") != "text" for item in content):
            return True
    return False


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    proxy: LlamaCacheProxy

    def do_GET(self) -> None:
        if _is_blocked_path("GET", self.path):
            self.send_error(404)
            return
        try:
            self.proxy.forward(self, "GET", _upstream_path("GET", self.path), b"")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def do_HEAD(self) -> None:
        if _is_blocked_path("HEAD", self.path):
            self.send_error(404)
            return
        try:
            self.proxy.forward(self, "HEAD", _upstream_path("HEAD", self.path), b"")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def do_OPTIONS(self) -> None:
        if _is_blocked_path("OPTIONS", self.path):
            self.send_error(404)
            return
        try:
            self.proxy.forward(self, "OPTIONS", self.path, b"")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def _send_upstream_error(self, error: Exception) -> None:
        LOGGER.exception("upstream request failed")
        if not getattr(self, "_proxy_response_started", False) and not self.wfile.closed:
            self.send_error(503, f"upstream unavailable: {error}")

    def do_POST(self) -> None:
        if _is_blocked_path("POST", self.path):
            self.close_connection = True
            self.send_error(404)
            return
        if not _is_known_post_path(self.path):
            self.close_connection = True
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True
            self.send_error(400, "Content-Length must be an integer")
            return
        if length < 0:
            self.close_connection = True
            self.send_error(400, "Content-Length must not be negative")
            return
        raw = self.rfile.read(length)
        upstream_path = _upstream_path("POST", self.path)
        if _is_non_evicting_post_path(self.path):
            try:
                self.proxy.forward(self, "POST", upstream_path, raw)
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        if _normalized_path(self.path) != "/v1/chat/completions":
            try:
                with self.proxy.foreground_operation():
                    self.proxy.prepare_uncached("before_unmanaged_post")
                    self.proxy.forward(self, "POST", upstream_path, raw)
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "request body must be JSON")
            return
        if not isinstance(body, dict):
            self.send_error(400, "request body must be a JSON object")
            return
        header_session_id = _session_id(self)
        body_session_id = _body_session_id(body)
        body = _without_proxy_affinity_fields(body)
        kv_channels = (
            _extract_kv_headers(self.headers),
            _extract_kv_body_fields(body),
            _extract_kv_tags(body, getattr(self.proxy, "inline_prefix", "")),
        )
        kv_controls = _resolve_kv_controls(
            getattr(self.proxy, "kv_controls", KvControls()), *kv_channels
        )
        if _has_media(body):
            try:
                with self.proxy.foreground_operation():
                    self.proxy.prepare_uncached("before_media")
                    self.proxy.forward(self, "POST", self.path, json.dumps(body).encode("utf-8"))
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        explicit_session_id = header_session_id or body_session_id
        if explicit_session_id is None and self.proxy.require_session_id:
            self.send_error(400, "a stable session ID is required")
            return
        session_id = explicit_session_id or _anonymous_session_id(
            body,
            getattr(self.proxy, "anonymous_affinity", "project"),
        )
        _log_event(
            "session_resolve",
            session_ref=_session_ref(session_id),
            source="header" if header_session_id else "body" if body_session_id else "anonymous",
        )
        plan = None
        reservation_released = False
        try:
            with self.proxy.session_operation(session_id):
                request, plan = self.proxy.begin_request(body, session_id, kv_controls)
                try:
                    result = self.proxy.forward(
                        self,
                        "POST",
                        self.path,
                        json.dumps(request).encode("utf-8"),
                    )
                    try:
                        # complete_request owns reservation release even if
                        # snapshot finalization raises.
                        reservation_released = True
                        self.proxy.complete_request(
                            plan,
                            result.status,
                            finish_reason=result.metadata.finish_reason,
                            cached_tokens=result.metadata.cached_tokens,
                            controls=kv_controls,
                        )
                    except Exception:
                        LOGGER.exception("response succeeded but snapshot finalization failed")
                finally:
                    if not reservation_released:
                        self.proxy.cancel_request(plan)
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def do_DELETE(self) -> None:
        self.close_connection = True
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("PI_LLAMA_CACHE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Generic (client-agnostic) settings use bare KV_* names; the optional
    # prefix exists for shared environments where those would collide.
    env_prefix = os.environ.get("LOCAL_LLM_KV_CACHE_ENV_PREFIX", "").strip()
    kv_controls = KvControls(
        save=_env_bool(f"{env_prefix}KV_SAVE", True),
        restore=_env_bool(f"{env_prefix}KV_RESTORE", True),
        save_growth=int(os.environ.get(f"{env_prefix}KV_SAVE_GROWTH", "0")),
        save_interval=int(os.environ.get(f"{env_prefix}KV_SAVE_INTERVAL", "0")),
        restore_tokens=int(os.environ.get(f"{env_prefix}KV_RESTORE_TOKENS", "0")),
    )
    proxy = LlamaCacheProxy(
        upstream=os.environ.get("PI_LLAMA_UPSTREAM", "http://127.0.0.1:8080"),
        cache_dir=os.environ.get("PI_LLAMA_CACHE_DIR", DEFAULT_CACHE_DIR),
        max_cache_gib=float(os.environ.get("PI_LLAMA_CACHE_MAX_GIB", "12")),
        wait_seconds=float(os.environ.get("PI_LLAMA_CACHE_WAIT_SECONDS", "120")),
        enable_prefix_seeding=_env_bool("PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING", True),
        enable_shared_prefixes=_env_bool("PI_LLAMA_CACHE_ENABLE_SHARED_PREFIXES", True),
        anonymous_affinity=os.environ.get("PI_LLAMA_CACHE_ANONYMOUS_AFFINITY", "project").strip(),
        prefix_seed_delay_seconds=float(os.environ.get("PI_LLAMA_CACHE_PREFIX_SEED_DELAY", "2")),
        prefix_seed_timeout_seconds=float(
            os.environ.get("PI_LLAMA_CACHE_PREFIX_SEED_TIMEOUT", "600")
        ),
        save_policy=os.environ.get("PI_LLAMA_CACHE_SAVE_POLICY", "all").strip().lower(),
        require_session_id=_env_bool("PI_LLAMA_CACHE_REQUIRE_SESSION_ID", False),
        shared_prefix_scope=os.environ.get("PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE"),
        minimum_shared_prefix_tokens=int(
            os.environ.get("PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS", "128")
        ),
        kv_controls=kv_controls,
        inline_prefix=os.environ.get("LOCAL_LLM_KV_CACHE_INLINE_PREFIX", "").strip(),
    )
    ProxyHandler.proxy = proxy
    host = os.environ.get("PI_LLAMA_CACHE_HOST", "127.0.0.1")
    port = int(os.environ.get("PI_LLAMA_CACHE_PORT", "8081"))
    server = ThreadingHTTPServer((host, port), ProxyHandler)
    LOGGER.info("listening on http://%s:%d -> http://%s:%d", host, port, proxy.upstream_host, proxy.upstream_port)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_on_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proxy.flush_dirty("shutdown")
        except Exception:
            LOGGER.exception("failed to flush dirty snapshots during shutdown")
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.server_close()


if __name__ == "__main__":
    main()
