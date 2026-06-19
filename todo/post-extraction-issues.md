# Post-extraction issues: protocol mismatch, token bug, test gaps

## 1. Gateway protocol mismatch (critical, blocks real E2E)

DirectClient speaks OpenAI chat completions format (`messages`, `choices[0].message.content`). The AI Gateway at `AI_COMPLETIONS_FURL` speaks a custom format:

- Request: `prompt` (string) or `input` (array of `{type, text}` parts), requires `metadata.service`
- Response: flat `{content, usage}`, not `{choices: [{message: {content}}]}`
- Tool calling: **not supported at all** -- gateway rejects `tools` in request body

All 10 slow integration tests fail with `400 Bad Request: "prompt or input is required"`.

Options:
- Add an OpenAI-compatible endpoint to the gateway (gateway change, not toolstream)
- Add a gateway adapter mode to DirectClient that translates between formats (toolstream change)
- Use a different OpenAI-compatible endpoint for tool-calling (Azure OpenAI direct, not through gateway)

This determines whether toolstream talks to the gateway or to Azure directly. The gateway currently only handles simple completions (text in, text out). Tool calling requires OpenAI-format chat completions.

## 2. Token double-counting bug (critical, production bug)

DirectClient emits **running totals** in StepFinish events (e.g., round 1: 100 tokens, round 2: 300 tokens). AsyncSession's `_send_direct` sums them via `+=` (100 + 300 = 400). The correct total for 2 API calls with 100 and 200 prompt tokens is 300, not 400.

The test `test_token_accumulation_across_steps` asserts 400 -- it passes but enshrines the double-counting as correct.

Fix options:
- Make DirectClient emit per-step deltas (not running totals), keep AsyncSession's `+=` -- this is the cleaner fix, each layer does simple accumulation
- Make AsyncSession take the last StepFinish value per turn (since it's the running total) -- simpler change but semantically odd

Either way, the test must be updated to assert the correct value.

Affected files:
- `toolstream/_direct.py` lines 125-132 (running total accumulation)
- `toolstream/_session.py` lines 54-60 (double accumulation via +=)
- `tests/test_integration.py` `test_token_accumulation_across_steps` (wrong assertion)

## 3. Mock integration test gaps

Found by independent critique of `tests/test_integration.py`:

### High priority
- `test_invoke_agent_with_message` does not verify tool definitions are sent to the API -- the most important part of invoke_agent
- `test_invoke_agent_with_message` replaces `_direct` after invoke_agent already created one, so it tests its own mock wiring rather than the actual invoke flow
- `test_multi_turn_conversation` does not verify conversation history accumulation (second API call should include first turn's messages)
- `test_multiple_tool_calls_in_single_response` has no token assertions on any StepFinish

### Medium priority
- `multi_tool_call_response` helper is defined in test_integration.py but should be in conftest.py alongside the other response helpers
- No test for tool handler exception during SyncSession (async path tested, sync queue bridge not)
- No test for unknown tool name in LLM response (hallucinated tool)
- No test for `tool_calls: None` vs missing `tool_calls` key (works by accident via `not None` being truthy)
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
