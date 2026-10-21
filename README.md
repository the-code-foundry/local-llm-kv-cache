# Local KV Cache for LLMs

The Code Foundry fork — [changes versus the original repo](./tcf.md).

This project adds a persistent KV cache to Hermes Agent and local OpenAI-compatible clients, notably Pi and Zed. It combines two mechanisms:

1. an in-memory hot cache, which directly reuses a llama.cpp slot to continue the same session;
2. persistent on-disk snapshots for sessions and stable prefixes, including a shared reference ("golden") prefix reusable by new sessions.

With the provided systemd unit, the proxy listens on `127.0.0.1:18082` and forwards requests to llama.cpp on `127.0.0.1:8080`. llama.cpp performs the actual prompt caching and checkpoint restoration; the proxy never fabricates false-positive results.

> [!IMPORTANT]
> Hybrid or recurrent models such as Qwen3.8 require a llama.cpp version that correctly persists their checkpoints. The validated deployment uses the `local/kv-restore-checkpoints` branch, commit `862535a`. Without this fix, llama.cpp may report `n_restored > 0` while the following request remains at `cache_n=0`. See [llama.cpp PR #26004](https://github.com/ggml-org/llama.cpp/pull/26004).

The detailed architecture and Mermaid diagrams are in [DESIGN.md](./DESIGN.md).

## Quick start

Prerequisites: Python 3.10 or later and an already-running llama.cpp server, with slot save/restore enabled and the hybrid checkpoint persistence fix applied.

```bash
mkdir -p ~/server-ops
git clone https://github.com/gbuisson/local-llm-kv-cache.git ~/server-ops/local-llm-kv-cache
mkdir -p ~/.config/systemd/user
cp ~/server-ops/local-llm-kv-cache/local-llm-kv-cache.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now local-llm-kv-cache.service
curl -fsS http://127.0.0.1:18082/health
```

If llama.cpp is not listening on `127.0.0.1:8080`, change `PI_LLAMA_UPSTREAM` in the user unit. Configure the Pi and Zed provider URLs as `http://127.0.0.1:18082/v1`.

### Production hardening options

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PI_LLAMA_CACHE_NAMESPACE` | `default` | Invalidates snapshots when the model, llama.cpp, context, or template changes. Use a stable deployment fingerprint. |
| `PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING` | `true` | Enables opportunistic seeding of shared reference prefixes. `-np 1` mode is supported through the transactional swap described below. |
| `PI_LLAMA_CACHE_ENABLE_SHARED_PREFIXES` | `true` | Enables shared-prefix discovery and restore. Set to `false` for session-snapshot-only backends that do not expose `/apply-template` and `/tokenize`, such as QwFNfer snapshot format v1. |
| `PI_LLAMA_CACHE_PREFIX_SEED_DELAY` | `2` | Delay in seconds between a successful response and the start of background seeding. The operation is deferred if a foreground request is waiting. |
| `PI_LLAMA_CACHE_PREFIX_SEED_TIMEOUT` | `600` | HTTP timeout reserved for the seeding `/completion`, in seconds. It must exceed the cold prefill time of the complete stable prefix. It does not change the 120-second timeout of other internal APIs and must be strictly positive. |
| `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE` | `default` | Isolation domain for shared prefixes. Personal and work environments must use distinct explicit values, for example `personal` and `work`. |
| `PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS` | `128` | Minimum length of the exact longest common prefix (LCP) required to retain a shared candidate. Must be a strictly positive integer. |
| `PI_LLAMA_CACHE_SAVE_POLICY` | `all` | `all` saves after every successful response. `terminal` defers the write for `finish_reason=tool_calls`, then saves after a terminal response, before eviction, or during a clean shutdown. |
| `PI_LLAMA_CACHE_REQUIRE_SESSION_ID` | `false` | When `true`, rejects cache-managed text chat requests that provide no session identifier. Requests containing media remain accepted: they are serialized but unmanaged after any dirty state has been saved. |
| `PI_LLAMA_CACHE_ANONYMOUS_AFFINITY` | `project` | Fallback key when no explicit session ID exists. `project` hashes only the stable system prefix; `conversation-start` also hashes the first non-system message so concurrent agent conversations do not share one snapshot. |
| `PI_LLAMA_CACHE_MAX_GIB` | `12` | LRU disk budget for snapshots. A successful restore or hot reuse updates recency. |

The `terminal` policy is intended for agents that chain many tool calls. Deferred state remains in the hot slot and is saved before a cold request can overwrite an owned slot. If slot metadata cannot be verified, the proxy takes the safe path and saves immediately.

### Per-request cache controls

Five generic controls decide whether and when this proxy reads and writes snapshots.
Each exists in four channels with the same key words; precedence is
**header > body field > inline tag > environment default**.

| key | type | env default | meaning |
| --- | --- | ---: | --- |
| `kv-save` | bool | `KV_SAVE=true` | Save a snapshot after this request. When `false`, nothing is written and the session state is not updated, so the next request restores the snapshot already on disk. |
| `kv-restore` | bool | `KV_RESTORE=true` | Restore a snapshot before this request. When `false`, every disk restore (session, shared prefix, legacy prefix) is skipped; hot slot reuse is unaffected. |
| `kv-save-growth` | int | `KV_SAVE_GROWTH=0` | Minimum token growth of the slot versus the snapshot on disk before the snapshot is rewritten. Skipped saves leave the state dirty, so eviction and shutdown still persist it. `0` always saves. |
| `kv-save-interval` | int | `KV_SAVE_INTERVAL=0` | Minimum seconds since the last persisted save of this session before the snapshot is rewritten. Skipped saves leave the state dirty, like `kv-save-growth`; both gates must pass for a save to happen. Without a recorded save time (e.g. the first save, or after a proxy restart) the proxy saves. `0` disables the gate. |
| `kv-restore-tokens` | int | `KV_RESTORE_TOKENS=0` | Minimum snapshot size, in tokens, worth restoring from disk. Applies where the token count is known: shared-prefix manifests, and session snapshots this proxy process has saved. `0` always restores. |

Channels, by example for `kv-save`:

- header: `X-KV-Save: false`
- body field: `"kv_save": false` (top level or inside `extra_body`; stripped before forwarding)
- inline tag: `[[x-kv-save: false]]` in message content
- environment: `KV_SAVE=false` (the deployment default)

Inline tags are recognized only in system messages and in the first non-system
message — the positions the conversation owner controls — and are stripped from
those positions before the request is forwarded, so the model never sees them and
the forwarded token stream stays identical whether a tag was present or not. Tags
in later messages are ordinary content and are forwarded untouched. Unknown
`x-kv-*` tags in scanned positions are stripped and logged. Invalid values in any
channel are logged and ignored, falling through to the next channel.

Two optional prefixes, both read once at startup:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `LOCAL_LLM_KV_CACHE_ENV_PREFIX` | empty | Prefix for the five `KV_*` variables on shared hosts where the bare names would collide. With `MYAPP_`, only `MYAPP_KV_SAVE` etc. are read. |
| `LOCAL_LLM_KV_CACHE_INLINE_PREFIX` | empty | Prefix for inline tags. With `b7-`, only `[[b7-x-kv-save: ...]]` is recognized and stripped; a bare `[[x-kv-save: ...]]` becomes ordinary content. This doubles as a weak capability: only clients that know the prefix can change cache behaviour from message content. Change it together with `PI_LLAMA_CACHE_NAMESPACE`. |

The typical fan-out pattern — compute one shared base conversation, then run
several independent tasks on top of it without the tasks overwriting the base
snapshot: send the base conversation with a session ID once (saved normally),
then send each `base + task` request with the same session ID and `kv-save:
false`. Each task restores the base snapshot, llama.cpp processes only the task
tokens, and the base snapshot stays untouched.

For long conversations, where rewriting the snapshot after every answer costs
seconds while holding the session lock, combine the two lazy-save gates:
`KV_SAVE_INTERVAL=300` with `KV_SAVE_GROWTH=2000` rewrites the multi-GB file at
most once per five minutes and only once the conversation has grown by 2,000
tokens; everything in between stays hot in the slot and is still persisted by
eviction and shutdown flushes. The trade: turns since the last save are lost if
llama-server dies without a flush, and are then re-processed from the older
snapshot.

### QwFNfer session snapshots

QwFNfer snapshot format v1 implements the same single-slot save/restore contract,
but deliberately supports exact session continuation only. Run the proxy on the
same host as QwFNfer, point `PI_LLAMA_CACHE_DIR` at the directory passed to
`qwfn-server --snapshot-dir`, and isolate its archive namespace from llama.cpp:

```ini
PI_LLAMA_UPSTREAM=http://127.0.0.1:18080
PI_LLAMA_CACHE_DIR=/var/lib/qwfnfer/snapshots
PI_LLAMA_CACHE_NAMESPACE=qwfnfer-qwen38-flash-next-q4-192k-v1
PI_LLAMA_CACHE_ENABLE_SHARED_PREFIXES=false
PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=false
PI_LLAMA_CACHE_SAVE_POLICY=terminal
PI_LLAMA_CACHE_REQUIRE_SESSION_ID=false
PI_LLAMA_CACHE_ANONYMOUS_AFFINITY=conversation-start
```

The QwFNfer archive contains its own model/configuration fingerprint and SHA-256
integrity envelope. It is not compatible with llama.cpp `.bin` slot files. A
restore is useful only when the next prompt is an exact extension of the saved
token sequence; QwFNfer resets and fully prefills on any divergence. The
`conversation-start` fallback covers clients such as BB ACP that do not
propagate a stable Hermes session ID. Exact-extension validation remains the
fail-closed guard if two conversations start with identical content.

### Concurrent slot safety

The proxy permits requests for different sessions to overlap when llama.cpp has
distinct idle slots. Requests for the same session remain serialized. Foreground
requests and background prefix seeds reserve their slots atomically, so neither
can restore, save, or evict a snapshot owned by the other. A single-slot server
therefore remains serialized, while a two-slot server can execute two independent
sessions concurrently without crossing snapshots. Pending prefix-seed work is
bounded and yields to foreground traffic.

### Production Hermes profile

The systemd template in this repository uses `127.0.0.1:18082` for the proxy and `127.0.0.1:8080` for llama.cpp. A Nix deployment may choose another explicit pair, for example proxy `:8080` and upstream `:8081`. In all cases, never run two proxies on the same port or two llama.cpp servers on the same port at the same time.

The validated personal Hermes profile is fail-closed for text:

```ini
PI_LLAMA_CACHE_REQUIRE_SESSION_ID=true
PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE=hermes-default-personal
PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=true
PI_LLAMA_CACHE_PREFIX_SEED_TIMEOUT=600
```

A text request without a session identifier then receives HTTP `400`. A request containing media remains outside cache management and is not persisted as a session. The client must provide stable affinity through one of the supported fields or headers; do not enable a generic `--pass-session-id` mechanism that would alter the client contract without validation.

The proxy does not change generation or reasoning policy. In particular, the validated llama.cpp profile retains:

```text
--predict 24576
--reasoning on
--reasoning-format deepseek
--reasoning-preserve
--reasoning-budget 16384
--reasoning-budget-message ". Thinking done, time to output the answer."
```

On the Hermes side, `--reasoning medium` is recommended. The `high` level is discouraged for this profile because the client mapping may reach `reasoning_effort=xhigh` and substantially increase cost without improving prefix caching.

Explicit affinity may come from `session_id`, `conversation_id`, or `prompt_cache_key`, either at the root of the JSON body or in `extra_body`. It may also come from the `X-Session-Affinity`, `X-Session-Id`, `X-Conversation-Id`, `X-Pi-Session-Id`, `X-OpenCode-Session`, or `X-Client-Request-Id` headers. The proxy removes these affinity fields before forwarding the request to llama.cpp.

### Public route contract

Routing is explicit and closed by default:

| Class | Routes | Slot behavior |
| --- | --- | --- |
| Session cache | `POST /v1/chat/completions` | Session-bound hot/disk KV lifecycle. |
| Utilities and control | `POST /tokenize`, `/detokenize`, `/apply-template`, token-counting routes, and `/v1/chat/completions/control` | Forwarding without forced KV save or eviction. |
| Unmanaged inference | Completions, responses, embeddings, infill, reranking, and Anthropic messages routes | Save dirty states, clear ownership, then forward. |
| Monitoring and model metadata | Ordinary `GET`, `HEAD`, and `OPTIONS` routes | Transparent forwarding without slot mutation. |
| Administration | `/slots`, LoRA/tool administration, mutable `/props`, model load, unload, or download, and all `DELETE` requests | `404` response, never forwarded. |
| Unknown `POST` | Any route absent from the allowlist | `404` response, never forwarded. |

For clients that systematically prefix llama.cpp utilities with `/v1`, the proxy translates `GET`/`HEAD /v1/props` to `/props`, and `POST /v1/tokenize`, `/v1/detokenize`, and `/v1/apply-template` to their native unprefixed routes. `HEAD` strips the upstream response body. `OPTIONS` and CORS preflight headers are forwarded unchanged.

### Streaming contract

The proxy relays responses progressively. It reads one available fragment from the upstream socket with `HTTPResponse.read1()`, immediately writes a valid HTTP/1.1 chunk, then calls `flush()` toward the client. Do not replace this path with `HTTPResponse.read(size)`: that method may wait for `size` bytes or the end of the stream and turn a token stream into a response delivered all at once.

The proxy filters headers named in its static `HOP_BY_HOP_HEADERS` set (`Connection`, `Content-Length`, `Keep-Alive`, `Proxy-Authenticate`, `Proxy-Authorization`, `TE`, `Trailer`, `Transfer-Encoding`, and `Upgrade`) from non-HEAD downstream responses, then emits `Transfer-Encoding: chunked`. `Content-Length` is preserved for HEAD responses. The proxy does not yet parse extension header names that may be advertised in the value of `Connection`; the upstream must therefore remain a trusted llama.cpp instance that does not emit them. SSE payloads, including `[DONE]`, remain byte-for-byte identical. Only a bounded tail of the stream is retained to extract completion metadata.

The `test_forward_streams_body_and_filters_hop_by_hop_headers` regression test verifies that multiple upstream fragments remain multiple downstream chunks and that the forwarding path calls `read1`, not `read`. In production, a timing probe must observe the first `data:` event before the last, with a non-zero time gap during a multi-token generation.

## Problem addressed

The first message from a coding agent often contains:

- the agent's system prompt;
- project rules and the contents of `AGENTS.md`;
- tool definitions and their JSON schemas;
- the model's chat template parameters.

These data change little within a project, but a new session would normally recompute them in full. The proxy separates the stable prefix from the dynamic part of the conversation to avoid this repeated prefill.

## Architecture

```text
Hermes Agent / Pi / pi-acp / Zed
          |
          v
127.0.0.1:18082  cache_proxy.py
          |
          +-- in-memory session/slot mapping
          +-- ~/.llama-slot-cache/*.bin
          +-- private *.bin.manifest.json manifests
          |   (scope + exact token IDs)
          |
          v
127.0.0.1:8080  llama.cpp llama-server
```

The provided template listens only on the loopback interface. A declarative deployment may choose an explicit LAN address, but it must then restrict access through a firewall or trusted private network: the proxy forwards authentication headers upstream but adds no authentication of its own. Never expose it directly to the Internet.

### Slot ownership and cache hierarchy

Session-to-slot affinity is kept in memory by the proxy. A request does not steal the complete context of another session to reuse its stable prefix. When no hot session is available, the proxy prefers an inactive unowned slot, then follows this hierarchy:

```text
hot session → session snapshot → shared golden prefix
→ exact historical prefix → cold request / full prefill
```

A session evicted from memory can therefore restore its own complete snapshot without treating another session's dynamic history as its cache.

The historical fallback is available only in the `default` scope. A scope other than `default` never uses a historical snapshot without a manifest. Even in `default`, the presence of the companion manifest blocks this fallback: a rejected shared pair cannot be retried as an unverified historical prefix.

## Cache keys

### Logical prefix key

The proxy first constructs an object representing the stable prefix. In `messages`, it retains contiguous messages with the `system` or `developer` role from the beginning and stops at the first `user`, `assistant`, or `tool` message.

It then adds the fields that affect prompt formatting:

```text
model
tools
tool_choice
chat_template_kwargs
chat_template_args
enable_thinking
reasoning_effort
reasoning_format
response_format
json_schema
grammar
add_generation_prompt
continue_final_message
parallel_tool_calls
```

The key is the SHA-256 of sorted canonical JSON:

```text
prefix_key = SHA256(canonical_json(prefix_payload))
```

The implementation is in [cache_core.py](./cache_core.py).

### Snapshot names

The filename also includes the cache version, its kind, and its identity:

```text
snapshot_key = SHA256(UTF8("3\0" + namespace + "\0" + kind + "\0" + identity + "\0" + prefix_key))
```

Produced formats:

```text
local-llm-session-<hash>.bin
local-llm-prefix-<hash>.bin
```

- For `session`, the identity is the Pi/Zed session identifier.
- For `prefix`, the identity is the fixed string `prefix`, allowing sessions from the same project to share the file.
- Version `3` and the deployment namespace invalidate older formats or incompatible caches.

User messages, assistant history, and tool results are not part of `prefix_key`. They form the dynamic part, handled by llama.cpp's common-prefix search.

## Shared golden prefix: tokens are authoritative

`prefix_key` is used for session snapshots and the exact historical fallback. To share a prefix across different seed operations, the proxy does not consider a "close" JSON hash to be proof. When a text request has neither a hot session nor a session snapshot, it computes the actual input with the APIs of the current llama.cpp instance:

1. `POST /apply-template` with the body of the future chat completion request;
2. retrieve the complete rendered prompt;
3. `POST /tokenize` with `add_special=false` and `parse_special=true`;
4. compare the token ID sequence with manifests on disk.

Each golden snapshot has a companion manifest:

```text
local-llm-prefix-<hash>.bin
local-llm-prefix-<hash>.bin.manifest.json
```

The manifest contains the schema version, `PI_LLAMA_CACHE_NAMESPACE`, `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE`, the snapshot basename, and the complete token ID sequence. The proxy considers only valid manifests from the same namespace and scope whose `.bin` file actually exists.

Among these candidates, it selects the longest exact common prefix (LCP) that reaches `PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS`. Comparison is token by token, with no hash similarity, percentage, or textual heuristic. At equal LCP length, a candidate fully contained in the current prompt wins over one that diverges at the next token.

### Strict restore condition

A golden candidate is restored only if its complete token tuple is a prefix of the current prompt:

```text
candidate_tokens == request_tokens[:len(candidate_tokens)]
```

The restore is then pinned to the request's slot. llama.cpp resumes computation from that state and processes the remaining suffix.

If the candidate diverges before its end, the proxy does not restore it, even if its LCP exceeds the threshold. The request runs cold. After a successful response, the proxy schedules background seeding of `request_tokens[:verified_lcp]`, whose tokens have already been verified. This behavior progressively turns an old prefix that is too long into a shorter stable prefix, without reusing state beyond the divergence.

Thus, a date, memory, skill, or user suffix changed late in the prompt still allows earlier tokens to be reused. An early change to the system prompt sharply shortens the LCP and may cause a completely cold run. Once a shorter prefix has been published, future compatible requests can restore it directly.

`PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE` is a data and trust boundary, not merely a performance setting. Personal and work environments must be isolated:

```ini
Environment=PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE=personal
# The work instance uses another unit and another scope:
Environment=PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE=work
```

Do not share the same scope between these environments. After an incompatible change to the model, GGUF, llama.cpp, context, or template, also change `PI_LLAMA_CACHE_NAMESPACE`.

## Request lifecycle

### 1. Session resolution

The proxy looks for affinity in this order:

1. `X-Session-Affinity`, `X-Session-Id`, `X-Conversation-Id`;
2. `X-Pi-Session-Id`, `X-OpenCode-Session`, `X-Client-Request-Id`;
3. `session_id`, `conversation_id`, or `prompt_cache_key` in the body or `extra_body`;
4. otherwise, `anonymous-<prefix_key>` if `PI_LLAMA_CACHE_REQUIRE_SESSION_ID=false`.

With `PI_LLAMA_CACHE_REQUIRE_SESSION_ID=true`, a text chat request without explicit affinity receives a `400` error: the mode fails closed instead of implicitly grouping conversations. A request with an image or other media bypasses session-cache management: the proxy acquires the global lock, saves dirty states, forgets slot owners, then forwards the request without a snapshot. This exception keeps multimodal requests working without assigning them misleading affinity.

For a hot or restored cache, the proxy adds only `id_slot`. The `cache_prompt` value provided by the client, or its default value in llama-server, is preserved. Disabling the native cache removes the LCP benefit but cannot cause stale state to be reused after a divergence.

### 2. Cache lookup

| Level | Condition | Action |
| --- | --- | --- |
| Hot session | Same session identifier, same `prefix_key`, inactive slot, and identical token count | Reuse the in-memory slot directly. |
| Free slot | No compatible hot session | Prefer an unowned slot to avoid evicting another active session. |
| Session snapshot | `local-llm-session-*.bin` exists and restoration succeeds | Restore the session's own context. |
| Shared golden prefix | Valid manifest in the same namespace/scope, complete candidate is a prefix of the request, threshold reached, and restoration succeeds | Restore the verified candidate. |
| Exact historical prefix | Scope is `default`, exact `local-llm-prefix-*.bin` file exists without a companion manifest, restoration succeeds, and no shared candidate was rejected | Compatibility with the old format, followed by scheduled migration to a verified manifest. |
| Cold cache | All previous conditions fail | Full prefill without a forced `id_slot`. |

A request bound to a hot or restored cache receives:

```json
{
  "id_slot": 1
}
```

A genuinely cold request receives no `id_slot`. Errors from `/apply-template` or `/tokenize`, malformed responses, invalid or orphaned manifests, incompatible scopes/namespaces, an LCP that is too short, or restore failure are never guessed around. The proxy proceeds with the strictly permitted historical fallback, otherwise cold, without turning a cache-discovery problem into a user-request failure.

### 3. Saving the session

With the default policy, a successful response atomically saves the current slot:

```text
current slot → local-llm-session-<hash>.bin
```

The proxy then updates the in-memory session/slot mapping.

With `PI_LLAMA_CACHE_SAVE_POLICY=terminal`, a response with `finish_reason=tool_calls` leaves dirty state in the hot slot. The next tool turn in the same session reuses it directly. The proxy saves this state after the terminal response, before any possible eviction, and during a clean shutdown. If it cannot verify slot metadata, it saves immediately.

### 4. Asynchronous prefix seeding

To create a stable prefix, the proxy obtains exact tokens through `/apply-template` followed by `/tokenize`, or reuses the already-verified LCP of a divergent candidate. It sends that sequence directly to `/completion`:

```text
prompt       = request_tokens[:verified_lcp]  # integer array, contents not logged
cache_prompt = false
n_predict    = 0
stream       = false
```

The snapshot is published only after an atomic save and verification that `n_saved == len(tokens)`. The identity proof remains the exact array sent to `/completion`; the saved count is an additional integrity check. The proxy then atomically publishes the companion manifest with mode `0600`. On restore, it also requires `n_restored == len(manifest.tokens)`; otherwise, it rejects the pair and attempts to clean it up. The cache directory uses mode `0700`.

A free, inactive, unowned slot is used first. With llama.cpp running as `-np 1`, seeding may borrow the only slot only if its owner is inactive and clean, its token count matches, and its complete session snapshot already exists. Under the global lock, the proxy:

1. saves the owner to a separate guard snapshot and verifies the token count;
2. computes and saves the prefix, then publishes its manifest;
3. restores the owner in a `finally` block and verifies its token count again before releasing the lock.

If this restore fails or returns an inconsistent count, the proxy forgets only hot ownership. The session's original persistent snapshot remains available for the next request.

The seeding queue is bounded to 16 entries. For the same target, it retains the longest verified work instead of replacing it with a shorter or unverified prefix. When the limit is reached, it removes the oldest entry that is not running. An old background task cannot remove a newer request: it removes the entry only if the stable tokens exactly match those it just processed.

Before any prefill, the proxy rescans valid manifests. If the namespace, scope, and complete token tuple already exist, it emits `prefix_seed_duplicate_skipped` and performs neither a second prefill nor a randomly named copy. This deduplication is exact.

A dirty owner, missing session snapshot, inconsistent token count, waiting foreground request, busy proxy, or API/save/manifest error causes seeding to be safely deferred or abandoned. This optimization remains opportunistic and does not participate in session-cache correctness. The first session pays its normal prefill cost, then possibly the cost of one additional background prefill and save.

Hybrid GDN models such as Qwen3.8 do not make it safe to treat a complete snapshot containing old user/assistant history as a universal prefix. The golden snapshot must therefore contain only the proven prefix.

## Context compaction and prefix evolution

A compaction boundary must not blindly extend the long previous transcript. Hermes Agent compacts in the existing session by default (`compression_in_place=True`), so provider-derived affinity remains unchanged. The proxy can reuse the hot slot or restore that session's latest snapshot after eviction. llama.cpp then computes its LCP against the compacted request, discards KV state after the divergence, and prefills the summary and protected tail.

Hermes may also create a child session with `boundary_reason="compression"`. The Hermes session identifier then produces new proxy affinity. The parent snapshot remains isolated; the child starts cold unless a compatible shared prefix exists.

If compaction changes the stable `system`/`developer` prefix, the hot cache no longer matches the `prefix_key`. The proxy saves any dirty state and starts again from compatible state or cold. Production validation must cover compaction on a hot slot and after eviction. In the latter case, llama.cpp—not the proxy—computes the LCP after restoring the session checkpoint and discards later KV state. This session-specific behavior never permits restoration of a divergent shared-prefix candidate.

Shared-prefix evolution follows the same rule:

- complete candidate is a prefix of the request: restoration allowed;
- divergent candidate: no restoration, cold request, then asynchronous seeding of the verified LCP;
- LCP below the threshold: cold request with no shared candidate;
- equal LCP: the complete candidate is preferred over the divergent one.

## This cache is not a response cache

The system stores:

```text
KV(prefix) + current slot state
```

It does not store:

```text
question → answer
```

The same system prompt means only that the model can resume from the same cached state. A different user suffix, tool result, sampling parameters, or randomness can still produce a different response.

Even for two identical requests, identical output should be expected only if the model version, inference parameters, `seed`, samplers, and hardware computation path are also identical. The validated llama-server does not set `--seed` and therefore uses a random seed by default. `temperature`, `top_p`, `top_k`, `seed`, and `max_tokens` are not part of `prefix_key` because they control generation, not the already-computed prompt state.

To test reproducibility, explicitly set the same `seed` and all sampling parameters, then compare the complete responses. `cached_tokens > 0` proves only that prompt tokens were reused.

## Invalidation and safe fallbacks

The following changes produce a new historical prefix key and shorten the golden LCP at the first actual divergence:

- text, order, spaces, or line breaks in the `system`/`developer` prompt;
- tools or tool JSON schemas;
- model identifier;
- thinking-mode or chat-template parameters;
- response format, grammar, or JSON schema;
- template behavior such as `add_generation_prompt` or `continue_final_message`;
- injected Hermes skills, instructions, or memory;
- date or other variable seeding data;
- actual output of the llama.cpp chat template or reasoning/thinking mode.

The disk cache may also be missed if:

- the session identifier changes, even if a shared prefix may still match;
- the snapshot was evicted by the 12 GiB limit;
- the file is damaged or llama.cpp refuses to restore it;
- no slot is available;
- the request contains media;
- the cache-format version changes.

A rendering or tokenization error, malformed response or manifest, orphaned manifest, inconsistent `n_saved`/`n_restored` count, or restore failure must not cause inference to fail. The proxy logs no prompt or token content; its pseudonymous operational metadata remains privacy-sensitive and requires the controls described below. In the historical `default` scope, it may still attempt the strictly permitted exact legacy fallback; in any other scope, or if that fallback is not admissible, it runs the prompt cold. Strict token comparison guarantees that a configuration change produces a shorter hit or a miss, never a restore beyond a divergence.

### Model change

`prefix_key` does not include the GGUF hash, the llama.cpp build hash, or every startup argument. After a change to the model, quantization, chat template, or incompatible KV configuration, change `PI_LLAMA_CACHE_NAMESPACE`. Never mix slot snapshots from different model versions.

## Structured telemetry and privacy limitations

Operational events are compact JSON objects. They never contain the raw session identifier, rendered prompt, token ID arrays, or manifest contents. `session_ref` is a truncated SHA-256 intended for correlation: it is a pseudonym, not cryptographic anonymization. A predictable source identifier could be recovered by a dictionary attack. Logs must therefore remain private, access-controlled, and subject to short retention. Shared-prefix events publish only counters, durations, controlled filenames, error types, and verified LCP lengths.

For a restored golden candidate, the useful sequence is:

```text
candidate: shared_prefix_candidate_restored(candidate_tokens, verified_lcp)
result   : shared_prefix_effective_hit(cached_tokens > 0)
rejected : shared_prefix_rejected_by_llama(cached_tokens == 0)
unknown  : shared_prefix_effectiveness_unknown
other    : cache_hit(layer=hot/session/prefix)
API      : usage.prompt_tokens_details.cached_tokens > 0
timings  : timings.cache_n > 0
```

`shared_prefix_candidate_restored` confirms only that a verified candidate was restored; proof of effective reuse comes from `shared_prefix_effective_hit`, `cached_tokens`, or `timings.cache_n`. A divergent candidate produces `shared_prefix_overlap_discovered`, with no restoration.

## Measured results

The redacted, versioned report [docs/validation/2026-09-14-production.md](./docs/validation/2026-09-14-production.md) ties the September 14, 2026 measurements to the commit, source fingerprints, validation commands, and deployment status. It distinguishes production evidence, deterministic tests, and historical measurements without publishing private logs or host-specific generation paths.

Measurements obtained with Qwen3.8 deployments and the patched llama.cpp:

- 32,185-token prompt: cold prefill in 54.8 s, then `cache_n=32151`, `prompt_n=34`, and 0.37 s after restoration in the same process;
- restore of a 12,700-token snapshot after a full llama.cpp restart: `n_restored=12700`, then `cache_n=12662`, `prompt_n=38`, and 0.356 s on a divergent prompt;
- on the actual proxy path, restore of a session evicted by another: `cached_tokens=12743/12793`;
- without the llama.cpp fix, the same scenario reports `n_restored` but remains at `cache_n=0`; the restore log alone therefore does not prove a hit;
- actual Hermes baseline measurement from September 12, 2026 on Qwen3.8 Q6_K, 128K context, `q4_0` KV, and checkpoints: 25,960 prompt tokens, `cached_tokens=0`, 233.94 s prompt evaluation, and 110.97 tokens/s; the session snapshot is 824,255,044 bytes (786.07 MiB);
- the most recent actual Hermes validations of the golden prefix effectively reused **24,464** and then **24,479** tokens, confirmed by response metadata, not merely by the restore call.

The main gain comes from avoiding prefill for the system prompt, project rules, and tool schemas. The number of avoided tokens appears directly in `cached_tokens` and `timings.cache_n`.

## Costs and limitations

- This optimization accelerates prompt prefill. It does not reduce weight VRAM usage or increase token/s decoding throughput.
- Snapshot size depends heavily on the model, context, and recurrent checkpoints. Older measurements ranged from 150 to 240 MB; the current checkpointed 128K snapshot reaches `786.07 MiB`. A 12 GiB budget can hold only about 15 snapshots of this size, with less room in practice for other lengths and manifests.
- Seeding is opportunistic. The first creation of a golden adds a prefill and a save; it is deferred if a request is waiting or if no safe slot swap is possible.
- The global lock serializes managed operations to prevent one snapshot from overwriting another context. Concurrent clients may wait for a free slot.
- Snapshots and manifests are private data. Manifest token IDs can be converted back with the same tokenizer: they are neither fingerprints nor anonymized data. Keep the local directory at mode `0700`, files at `0600`, do not synchronize them, and do not expose the proxy port.

## Operations and diagnostics

Service status:

```bash
systemctl --user status local-llm-kv-cache.service
systemctl --user is-active local-llm-kv-cache.service
journalctl --user -u local-llm-kv-cache.service -n 100 --no-pager
```

Health checks:

```bash
curl -fsS http://127.0.0.1:18082/health
curl -fsS http://127.0.0.1:18082/v1/models
```

### Deployment and rollback

Before a deployment:

1. record the candidate commit/package and the currently active generation as the rollback target;
2. inventory, without displaying them, the counts of prefix snapshots, manifests, and sessions, as well as free space;
3. run Python compilation, all 123 tests, branch coverage, and `git diff --check`;
4. build the package and declarative configuration without stopping the active service;
5. perform a dry activation when the system manager supports it;
6. activate exactly one proxy instance and one llama.cpp instance, then verify units, ports, and `/health`;
7. run a streaming probe and a real Hermes A/B test before declaring the deployment successful.

For a NixOS deployment, retain the full path of the previous generation. Rollback must not delete the cache:

```bash
sudo /nix/store/<previous-generation>/bin/switch-to-configuration switch
```

After rollback, verify services, ports, and `/health` again. If only seeding is causing problems, the minimal fallback is to declaratively set `PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=false` again and then reactivate the previous configuration. Temporarily bypassing the proxy is possible to isolate a failure, but loses affinity and restores; it does not justify any automatic snapshot purge.

Deleting golden prefixes is a separate maintenance operation. It must never be implicitly incorporated into a deployment or rollback.

Inventory manifests without displaying their reversible tokens:

```bash
python3 - <<'PY'
import json
from pathlib import Path

d = Path.home() / ".llama-slot-cache"
for p in sorted(d.glob("*.manifest.json")):
    try:
        m = json.loads(p.read_text())
        print(
            p.name,
            "namespace=", m.get("namespace"),
            "scope=", m.get("scope"),
            "tokens=", len(m.get("tokens", [])),
            "snapshot_exists=", (d / str(m.get("snapshot"))).is_file(),
        )
    except Exception as e:
        print(p.name, "MALFORMED", type(e).__name__)
PY
journalctl --user -u local-llm-kv-cache.service --since '10 minutes ago' --no-pager \
  | grep -E 'shared_prefix|cache_hit|cache_miss|prefix_seed'
```

### Targeted prefix cleanup

> [!CAUTION]
> This operation is destructive. It must never be used as a general cleanup command. No existing golden may be deleted without an inventory, retention criteria, a dry run, and explicit operator confirmation. Session snapshots must remain intact.

Step 1, dry inventory only:

```bash
python3 - <<'PY'
from pathlib import Path

d = Path.home() / ".llama-slot-cache"
files = sorted({*d.glob("local-llm-prefix-*.bin"), *d.glob("local-llm-prefix-*.bin.manifest.json")})
print("NO DELETION — prefix inventory:")
for p in files:
    print(p, "bytes=", p.stat().st_size, "mtime_ns=", p.stat().st_mtime_ns)
print(f"Total: {len(files)} file(s), {sum(p.stat().st_size for p in files)} bytes")
PY
```

Step 2, **manually** establish a retention allowlist and a deletion list. Retain at least:

- all session snapshots;
- all golden prefixes in the active namespace and scope whose usefulness has not been disproved;
- at least one snapshot/manifest pair for each exact token sequence still required;
- any artifact whose scope, provenance, or last-use date remains uncertain.

A duplicate may be declared only by exact comparison of its tokens in the same namespace and scope. An identical file size or hash is not sufficient to establish the same usefulness. Then place only approved basenames in `prefixes-to-delete.txt`, one `.bin` filename per line; never put a session snapshot on this list.

Step 3, dry-run the explicit list, without deletion:

```bash
python3 - <<'PY'
from pathlib import Path

d = Path.home() / ".llama-slot-cache"
names = [line.strip() for line in Path("prefixes-to-delete.txt").read_text().splitlines() if line.strip()]
targets = []
for name in names:
    if Path(name).name != name or not name.startswith("local-llm-prefix-") or not name.endswith(".bin"):
        raise SystemExit(f"Rejected name: {name!r}")
    snapshot = d / name
    manifest = d / f"{name}.manifest.json"
    if not snapshot.is_file():
        raise SystemExit(f"Missing snapshot: {snapshot}")
    targets.extend([snapshot, manifest] if manifest.is_file() else [snapshot])
print("DRY RUN — no deletion:")
for p in targets:
    print(p, "bytes=", p.stat().st_size)
print(f"Targeted total: {len(targets)} file(s), {sum(p.stat().st_size for p in targets)} bytes")
PY
```

Step 4, after reviewing the dry run, backing up the targeted files, and receiving explicit confirmation, stop the proxy and delete **exactly** this list. Then verify the counts of golden prefixes, manifests, and sessions, free space, service restart, and `/health`. Retain the backup archive until rollback has been validated.

Never broaden the patterns to `local-llm-*.bin`: doing so would also delete session snapshots. Do not turn the validated list into a glob at deletion time either.

### Golden-cache A/B procedure

Use real Hermes requests rather than the restore API result alone. Separate two validations: a complete golden hit, then convergence in the presence of an internal divergence.

```bash
# A: temporarily disable seeding in the declarative configuration,
# apply the configuration, then send a brand-new Hermes session.
PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=false

# B: enable an isolated A/B scope in the same declarative configuration,
# send an initial seeding session, wait for it to finish,
# then send a second new session.
PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=true
PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE=personal-ab-v1
PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS=128
```

For the complete hit, first confirm that A returns `cached_tokens=0`. In the isolated B scope, send an initial session to publish a golden, then a new session with the same stable prefix and a different user suffix. It must produce `shared_prefix_candidate_restored`, followed by `shared_prefix_effective_hit`, with a positive `cached_tokens`/`timings.cache_n` value and improved TTFT.

For adaptive convergence, make a controlled change to an element **inside** the old prefix, for example a skill or metadata injected late. The first pass must produce `shared_prefix_overlap_discovered`, must not restore the divergent candidate, and runs cold. After publishing the golden limited to the verified LCP, another variant compatible with that complete LCP must produce `shared_prefix_candidate_restored`, followed by `shared_prefix_effective_hit`. An early change must result in a shorter LCP or a cold run. Do not use a toolset change as proof of this branch without verifying the namespace: some clients then construct a separate cache space and produce a simple miss.

After the test, restore the final declarative `personal`/`work` scopes and reapply the configuration. Do not leave an imperative override in systemd.

## Tests and coverage

```bash
cd ~/server-ops/local-llm-kv-cache
python3 -m unittest -v test_cache_core.py test_cache_proxy.py
python3 -m py_compile cache_core.py cache_proxy.py
nix shell nixpkgs#python3Packages.coverage --command sh -c \
  'coverage run --branch --source=. -m unittest discover -s . -p "test_*.py" -v && coverage report --include="cache_core.py,cache_proxy.py" --fail-under=100'
```

At present, the suite contains 123 tests. It reaches 100% line and branch coverage on the modified `cache_core.py` and `cache_proxy.py` modules.

## Files and configuration points

- Pi: `~/.pi/agent/models.json`
- Zed: `~/.config/zed/settings.json`
- systemd template: [local-llm-kv-cache.service](./local-llm-kv-cache.service), to install at `~/.config/systemd/user/local-llm-kv-cache.service`
- Proxy: [cache_proxy.py](./cache_proxy.py)
- Keys and manifests: [cache_core.py](./cache_core.py)
- Disk cache: `~/.llama-slot-cache`

## License

Apache-2.0. See [LICENSE](./LICENSE).
