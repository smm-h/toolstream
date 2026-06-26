# Post-extraction issues: protocol mismatch, token bug, test gaps

## 1. Gateway protocol mismatch -- PARTIALLY RESOLVED

**Status:** Partially resolved. Toolstream now supports `auth_style="bearer"` (OpenAI) and `auth_style="x-api-key"` (gateway). The mismatch only matters for the gateway's custom protocol (non-OpenAI message format). Shopkeep's orchestrator uses OpenAI directly, bypassing the gateway entirely. The gateway path still uses a custom format (flat `{content, usage}` instead of `{choices: [{message: {content}}]}`), so direct gateway usage for tool-calling remains unsupported.

Original issue: DirectClient speaks OpenAI chat completions format (`messages`, `choices[0].message.content`). The AI Gateway at `AI_COMPLETIONS_FURL` speaks a custom format:

- Request: `prompt` (string) or `input` (array of `{type, text}` parts), requires `metadata.service`
- Response: flat `{content, usage}`, not `{choices: [{message: {content}}]}`
- Tool calling: **not supported at all** -- gateway rejects `tools` in request body

Remaining gap: if a consumer wants to use toolstream through the gateway (not OpenAI directly), they would need a gateway adapter or an OpenAI-compatible endpoint added to the gateway.

## 2. Token double-counting bug -- DONE

**Fixed in toolstream 0.3.0.** DirectClient now emits per-step deltas (not running totals) in each StepFinish event, and AsyncSession accumulates them via a single yield site with `+=`. The test `test_token_accumulation_across_steps` now correctly asserts the sum of per-step values (300 input = 100 + 200, 130 output = 50 + 80).

## 3. Mock integration test gaps -- PARTIALLY ADDRESSED

8 retry tests added (test_retry.py), 12 history tests added (test_history.py), 2 integration tests fixed. Reassessed gaps below.

### Resolved
- Unknown tool name in LLM response: covered by `test_dispatch_unknown_tool` in test_direct.py
- `tool_calls: None` vs missing key: covered by conftest `text_response` helper (always includes `"tool_calls": None`)

### Remaining -- high priority
- `test_invoke_agent_with_message` does not verify tool definitions are sent to the API (captures requests but does not check `tools` key in the request body)
- `test_invoke_agent_with_message` replaces `_direct` after invoke_agent already created one, so it tests its own mock wiring rather than the actual invoke flow
- `test_multi_turn_conversation` does not verify conversation history accumulation (verifies per-turn token counts but not that the second API call includes the first turn's messages)
- `test_multiple_tool_calls_in_single_response` has no token assertions on any StepFinish

### Remaining -- medium priority
- `multi_tool_call_response` helper is defined in test_integration.py but should be in conftest.py alongside the other response helpers
- No test for tool handler exception during SyncSession (async path tested, sync queue bridge not)
- `test_tool_error_handling` does not verify the error message was sent back to the LLM in the messages list

## 4. Shopkeep credential bugs (separate project, file a todo there)

These were found during the investigation and belong in shopkeep's todo, not toolstream's. Listed here for cross-reference:

### Critical
- `entrypoint.py` does not validate `ai_api_key` (empty string propagates silently, while `ai_base_url` gets `sys.exit(1)`)
- `crawl_pipeline.py` has `api_key: str = ""` and `base_url: str = ""` defaults -- foot-gun for onboard mode

### High
- `GITHUB_TOKEN` missing from `_ENV_BLOCKLIST` in `monitored_bash.py` -- LLM agent can extract it via `echo $GITHUB_TOKEN`
- GitHub token written to `~/.gitconfig` in plaintext via `entrypoint.py`
- `supabase_key` used without validation in `_call_enrichment`
- Hardcoded Lambda URL in `_normalize_samples` instead of env-based resolution

### Medium
- Env var precedence inconsistency: entrypoint.py prefers `AI_GATEWAY_API_KEY`, config.py prefers `SHOPKEEP_AI_GATEWAY_API_KEY`
- Duplicated normalizer URL resolution between entrypoint.py and scoring/pipeline.py
- `SpawnContext` has no `__post_init__` validation -- accepts empty credentials
- `_call_enrichment` silently swallows all exceptions (fire-and-forget)
- `ShopkeepContext` uses `Any` types, losing type safety

### Structural proposals
- Create `LLMCredentials` validated frozen dataclass (validates non-empty on construction) -- eliminates empty credential propagation, asymmetric validation, and env var precedence issues
- Single `resolve_llm_credentials()` function in config.py used by both entrypoint and CLI
- Mode-gated credential requirements: `run_pg_crawl` raises early in onboard mode if credentials are missing
- Flip `_ENV_BLOCKLIST` to an allowlist pattern -- only pass env vars the subprocess needs
- Centralize normalizer URL resolution -- delete the hardcoded Lambda URL and duplicated resolver functions
