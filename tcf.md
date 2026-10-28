# The Code Foundry fork — changes versus the original repo

This repository is The Code Foundry's fork of
[gbuisson/local-llm-kv-cache](https://github.com/gbuisson/local-llm-kv-cache),
forked at upstream `v0.2.0` (commit `2ffb14c`).

## Changes

Newest first.

### Request cancellation on client disconnect (added 2026-10-29)

The proxy previously noticed a client hang-up only when a write to the client
failed. A non-streaming request writes nothing until the whole answer is ready,
so a client timing out (e.g. behind llama-swap) left llama-server generating
until completion or the proxy's 1200 s upstream timeout, holding the slot and
the session lock the whole time. Now:

- every forwarded request runs a watcher that polls the client socket every
  250 ms; on end-of-file it shuts down the upstream connection, which wakes the
  blocked response wait immediately and makes llama-server stop generating
  (its own client — the proxy — disconnected);
- a cancelled request is abandoned, not completed: the reserved slot is
  released, no snapshot is written, and a `client_disconnected` event is
  logged;
- a request whose client is already gone is also abandoned at the
  reserve-a-slot loop, before it triggers a restore or waits up to
  `wait_seconds` for a slot;
- bytes arriving on the client socket (a pipelined follow-up request) count as
  alive; only EOF counts as a hang-up.

### Strata engine support (`PI_LLAMA_CACHE_ENGINE_HOLDS`, added 2026-10-25)

Support for engines whose own conversation cache keeps the last N conversations
in RAM and resumes them by prompt match — Strata — and whose slot metadata
cannot prove which conversation a slot holds (they do not report per-slot token
counts the llama.cpp way). `PI_LLAMA_CACHE_ENGINE_HOLDS=N` (default `0` = off,
matching llama.cpp behavior) makes the proxy:

- trust that the engine still holds the last N conversations it served since
  the proxy started, sending such sessions straight through without a disk
  restore or a `/slots` token-count check;
- save the live conversation before a different session takes the engine, so a
  deferred (`terminal`-policy) state survives the switch;
- forget the live conversation when a disk restore replaces it, because the
  restore overwrites the engine state without parking it.

Sessions beyond the newest N are dropped from proxy memory (logged as
`engine_hold_dropped`); their snapshots on disk remain and are restored when
they return. See "Engines that hold conversations (Strata)" in the README.

### Per-request cache controls (`kv-save`, `kv-restore`, `kv-save-growth`, `kv-save-interval`, `kv-restore-tokens`)

Five generic controls over snapshot reads and writes, each available as a
header (`X-KV-Save`), a body field (`kv_save`), an inline message tag
(`[[x-kv-save: false]]`) and an environment default (`KV_SAVE`); precedence
header > body > tag > env. See "Per-request cache controls" in the README.

- `kv-save: false` makes a request disposable: nothing is written, the session
  state is not updated. Built for fan-out — restore one shared base snapshot
  for several independent tasks without any of them overwriting it.
- `kv-restore: false` skips all disk restores for one request.
- `kv-save-growth` is a port of upstream's
  `fix/prune-tmp-and-raise-cache-limit` idea (there:
  `PI_LLAMA_CACHE_MIN_SAVE_GROWTH`) onto the multi-slot main: skip rewriting a
  multi-GB snapshot until the slot has grown by this many tokens. Divergences
  from that branch: renamed to the generic `KV_SAVE_GROWTH`, and a skipped
  save marks the session state dirty, so hot reuse keeps working and eviction,
  shutdown and `before_eviction` flushes still persist the unsaved tokens.
  (The branch's other fix, pruning orphaned `.tmp` snapshots, is already in
  upstream main.)
- `kv-save-interval` (added 2026-10-22) skips rewriting the snapshot until
  this many seconds have passed since the session's last persisted save. It
  combines with `kv-save-growth`: a save happens only when every enabled gate
  passes. Skipped saves mark the state dirty like a growth skip; a session
  without a recorded save time (first save, proxy restart) saves.
- `kv-restore-tokens` skips disk restores of snapshots smaller than this many
  tokens, where the token count is known (shared-prefix manifests, and session
  snapshots saved by the running process).
- `LOCAL_LLM_KV_CACHE_ENV_PREFIX` namespaces the `KV_*` env vars on shared
  hosts; `LOCAL_LLM_KV_CACHE_INLINE_PREFIX` namespaces the inline tags and
  doubles as a weak capability for who may control caching from content.
- Inline tags are recognized only in system messages and the first non-system
  message, and are stripped before forwarding, so the forwarded token stream —
  and therefore prefix matching — is identical with or without tags.

### Cherry-picked from upstream `feature/qwfnfer-session-snapshots`

Two upstream commits not yet in upstream main (`f42a76b`, `0a82133`):
`PI_LLAMA_CACHE_ENABLE_SHARED_PREFIXES=false` for session-snapshot-only
backends, and `PI_LLAMA_CACHE_ANONYMOUS_AFFINITY=conversation-start`, which
keys anonymous sessions on the stable prefix plus the first non-system message
so concurrent conversations that share a system prompt do not fight over one
snapshot.
