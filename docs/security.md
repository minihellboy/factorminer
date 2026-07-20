# Security Considerations

This document is the threat model and mitigation record for FactorMiner's
externally-facing surfaces, written after the Round 2 landscape-extension pass
(`docs/landscape-and-extensions.md` §10) added several genuinely new attack
surfaces: outbound data connectors, an authenticated network transport, and
LLM-authored content flowing into rendered reports and (optionally) back into
future prompts. It complements, rather than replaces, the existing guardrail
language in `docs/financial-services-integration.md` ("research artifacts
staged for review... does not recommend trades, size positions, bind risk, or
execute anything").

Every control below was verified against the current code, not just asserted
— see "Verification" under each surface for the concrete check performed.

## 1. Outbound data connectors (EDGAR, futures, crowding/consensus panels)

**Surface:** `data/edgar_source.py` (SEC EDGAR XBRL), `data/futures_source.py`
(continuous futures panels, offline/mock-only today), `evaluation/crowding.py`
(`ConsensusFactorPanel` — Ken French / AQR public factor-return CSVs),
`data/mcp_source.py` (pre-existing, Round 1).

**Threat model:** these are the only code paths that parse content FactorMiner
did not generate itself and does not control the source of. A malicious or
compromised upstream response could attempt to (a) trigger a crash / resource
exhaustion (oversized or malformed payload), (b) achieve code execution via
unsafe deserialization, (c) poison downstream research artifacts with
fabricated-but-plausible-looking financial data, or (d) exfiltrate data via a
redirected/spoofed endpoint.

**Mitigations (verified):**
- HTTPS-only. `ConsensusFactorPanel.fetch` explicitly refuses non-HTTPS URLs
  (`source=refused-non-https`, confirmed by test and by reading
  `evaluation/crowding.py`). EDGAR access is hardcoded to `https://data.sec.gov`.
- Explicit timeouts and response-size caps on every fetch (EDGAR: 30s / 8 MiB;
  crowding panel: configurable, default 4 MiB). Grepped the full diff for
  `urlopen(`/`requests.get(`/`requests.post(` calls without a `timeout=` —
  zero matches.
- `Content-Type` / shape validation before parsing; no `eval`/`exec`/
  `pickle.loads` anywhere in the new connector code (grepped the full round-2
  diff — zero matches for real `eval(`/`exec(` calls; the one `eval(` hit is
  PyTorch's unrelated `model.eval()` inference-mode switch).
- **Fail-closed, not fail-open, on malformed/adversarial data.** Verified by
  test: empty bytes, a truncated ZIP, a header-only CSV, ragged rows, and
  garbage text all parse to an empty series; `consensus_overlap_score` then
  reports `available=False` / `label=unavailable` rather than fabricating a
  falsely-reassuring "low crowding" score. This matters specifically because a
  silently-wrong crowding score is worse than an explicit "couldn't compute
  one" — a fail-open bug here would produce a confidently wrong research
  artifact.
- SEC's own fair-access policy is honored, not just avoided: a descriptive
  `User-Agent` including a contact email (default `FactorMiner Research Bot
  1.0 (contact@factorminer.local)`, override via `--user-agent` for
  production use) and a rate limiter capped at ≤10 requests/second (default
  8 rps) per `sec.gov/os/accessing-edgar-data`. **Verified live**: a real
  `attach-edgar` invocation against the actual `data.sec.gov` API during this
  review succeeded and returned genuine Apple Inc. XBRL facts, joined
  correctly onto a synthetic OHLCV panel by point-in-time filed date.
- Point-in-time discipline: EDGAR facts are forward-filled from their *filed*
  date, never their covered period-end — verified by a dedicated test
  asserting a fact filed 2024-02-15 is `NaN` before that date and populated
  only from that date forward, i.e. the connector cannot leak future
  information into a backtest.
- Local caching is keyed by a sanitized CIK (10-digit, validated), not by any
  free-form externally-influenced string — no path-traversal surface.

**Residual risk:** none of these connectors are wired into `mine`/`helix` by
default; a user must explicitly run `attach-edgar`/`build-futures` and point a
config at the resulting file. There is no automatic, unattended outbound
fetch triggered by normal mining.

## 2. MCP server: stdio (default) and opt-in HTTP transport

**Surface:** `factorminer/mcp/server.py`, `factorminer mcp-serve`.

**Threat model:** the MCP server is a tool-calling surface for an external
LLM/agent orchestrator. The two risks are (a) a network-facing listener with
weak or absent authentication, and (b) a tool whose description or output is
ambiguous enough that a calling agent misuses it (Anthropic's own published
finding: "bad tool descriptions dominate failures").

**Mitigations (verified):**
- **stdio remains the default transport** and requires no authentication by
  design — it is a local subprocess pipe, not a network listener. This is
  unchanged from Round 1.
- **HTTP transport is opt-in only** (`--transport http`), binds to
  `127.0.0.1` by default (grepped: the only occurrences of the literal string
  `0.0.0.0` anywhere in the round-2 diff are a code comment and a docstring
  explicitly stating it is *never* the default, plus a test asserting
  `DEFAULT_HTTP_HOST != "0.0.0.0"`), and **refuses to start** unless a
  non-empty bearer token is present in `$FACTORMINER_MCP_TOKEN` (or the env
  var named by `--auth-token-env`). **Verified live**: running
  `mcp-serve --transport http` with the token unset produces `Error: Refusing
  to start HTTP MCP transport without auth...` and a clean, non-zero-noise
  exit — not a silent unauthenticated listener.
- Auth is implemented against the real `mcp` SDK's `TokenVerifier`/
  `AccessToken` protocol (`StaticBearerTokenVerifier` in `mcp/server.py`),
  not a hand-rolled check bypassable by a client that skips it.
- Every `@mcp.tool()` docstring now documents exact argument/return shapes and
  ends with an explicit line that the output is a research artifact only,
  never an execution instruction — this is a structural guardrail (repeated
  on every tool) rather than a single paragraph in a doc a caller might not
  read.
- `inspect_debate` (new) and every other tool remain read-only: they invoke
  the FactorMiner CLI as a subprocess with an explicit argument list (never
  `shell=True`, confirmed by grep) and return structured data — no tool
  triggers an external side effect (a trade, an order, a filesystem write
  outside the caller-specified output directory).
- **Verified live** with a real MCP client (`mcp.client.stdio.stdio_client` +
  `ClientSession`, not a mock): listed 17 tools, confirmed the research-only
  guardrail text is present in a tool description, and executed a real
  `list_fsi_connectors` tool call end-to-end successfully.

**Residual risk:** the bearer-token scheme is a single static shared secret,
appropriate for a single-operator or trusted-network deployment; it is not a
substitute for a full OAuth/mTLS setup if FactorMiner's MCP server is ever
exposed beyond a trusted network. This is noted in
`docs/financial-services-integration.md`'s new LangGraph-consumption section.

## 3. Local-LLM cascade routing (`OpenAICompatibleProvider`)

**Surface:** `agent/llm_interface.py`.

**Threat model:** an OpenAI-compatible `base_url` option is, by construction,
a place the process will send HTTP requests. If that URL were ever attacker-
or remote-input-influenced, it becomes a Server-Side Request Forgery (SSRF)
primitive (e.g. pointed at a cloud metadata endpoint or an internal service).
A second risk is credential leakage: forwarding a real frontier-provider API
key to an untrusted custom endpoint.

**Mitigations (verified):**
- `base_url` is accepted only from the local YAML config object FactorMiner's
  own operator controls (`llm.cascade_draft_base_url` / equivalent), never
  from a request, a mined formula, an LLM response, or any other
  runtime-influenced value. Grepped and read the constructor: it raises
  `ValueError` if `base_url` is falsy rather than defaulting to something
  that could be silently wrong.
- **`OpenAICompatibleProvider` intentionally does not fall back to
  `OPENAI_API_KEY`** (verified by reading the code and by the existing test
  `test_openai_compatible_does_not_read_openai_api_key`) — a real frontier
  credential can never leak to a custom local/base_url endpoint even by
  accident.
- Frontier-provider client construction explicitly strips `base_url` before
  building the frontier half of a cascade pair, so a cascade config typo
  cannot redirect a real Anthropic/OpenAI-authenticated call to a third
  party.
- Explicit request timeouts on both the local/draft path (default 60s) and
  the frontier path (default 120s).
- Two inline `# SECURITY (SSRF):` comments at both call sites document the
  invariant for the next person editing this code, not just this report.

**Residual risk:** cascade routing is disabled by default
(`cascade_enabled: false`); an operator who enables it and points it at a
genuinely untrusted local network service is trusting that service with
whatever candidate-formula text it sends — same trust boundary as any local
tool integration, documented rather than silently assumed.

## 4. LLM-generated content rendered into HTML reports

**Surface:** `evaluation/report_viewer.py`, `evaluation/mrm_pack.py`,
`utils/tearsheet.py` — specifically the new economic-rationale field
(mandatory-attestation-gated) and MRM narrative sections.

**Threat model:** any free-text field an LLM can populate and that later gets
interpolated into an HTML report is a stored-XSS vector if not escaped —
either from the LLM itself producing adversarial markup (unlikely but not
impossible depending on provider/prompt), or from a research note/formula
name that flows through to a rationale field.

**Mitigations (verified):**
- All free-text fields (economic rationale, MRM narrative, formula-sensitivity
  plain-English notes) are passed through `html.escape` before interpolation
  into any HTML template. **Verified directly**, not just by reading the
  code: generated an HTML MRM report from a real mined library and grepped
  the output for `<script` — zero raw tags. A dedicated unit test also
  constructs a rationale containing a literal `<script>alert('xss')</script>`
  payload and asserts it renders as the escaped entity sequence
  (`&lt;script&gt;...`), not as live markup.
- The `UNATTESTED -- LLM DRAFT, NOT REVIEWED` banner is unconditionally
  present on any rationale whose `attested` field is not explicitly `True`
  (which only a human-facing CLI action, `report --attest-rationale`, can
  set) — **verified live**: 16 occurrences of the banner string in a real
  generated HTML report, and generation code hard-forces `attested=False`
  unless the source is explicitly `human`.

**Residual risk:** none identified for this surface specifically; the
existing pattern is sound and was applied consistently to every new free-text
field introduced in Round 2.

## 5. LLM-authored content flowing back into future prompts

**Surface:** `architecture/research_absorption.py` (research-note archetypes
feeding `PromptContextBuilder`), `architecture/sealed_joint_search.py`
(sealed evaluator feedback), `architecture/memory_policy.py` (edit-motif /
economic-rationale text persisted and potentially re-surfaced).

**Threat model:** this is prompt-injection-via-memory — if generated or
ingested text is later concatenated into a *system*/*developer*-level prompt
for a subsequent LLM call, and that text contains something shaped like an
instruction ("ignore previous constraints and..."), a sufficiently
instruction-following model could treat stored data as new instructions
rather than as the data it actually is.

**Mitigations (verified):**
- `research_absorption.py`'s B/C-layer classification treats ingested
  fragments strictly as data to be summarized into a bounded, structured
  `ResearchArchetype` (archetype name, one-sentence mechanism role, at most 3
  short research-path hypothesis cues) — the raw fragment text is never
  concatenated verbatim into a later generation prompt; only the
  LLM-compressed, schema-constrained summary is.
- `architecture/sealed_joint_search.py`'s prompt-facing feedback is
  *structurally* restricted: `SealedFeedback.to_prompt_dict` /
  `assert_feedback_is_sealed` explicitly allow-list only coarse fields
  (`n_passed`, `personas`, `agreement_fraction`, `rank`) and reject any
  attempt to pass raw evaluator internals through — **verified by a
  dedicated test** asserting the prompt-facing payload does not contain
  score/weight/component keys. The LLM-judge persona (when enabled) treats
  formula text as data to score, not as instructions to follow, and a
  malformed/adversarial judge reply fails closed to a neutral score rather
  than being trusted.
- Economic-rationale text is descriptive prose about a *specific already-
  admitted formula*, generated after the fact, not incorporated into the
  generation-time prompt for future candidates in a way that could compound.

**Residual risk:** this is a defense-in-depth posture (schema constraints,
allow-listed fields, fail-closed judges), not a formal proof that no LLM
provider could ever be confused by adversarial input. No known concrete
exploit exists against this codebase today; the mitigations above are the
correct, standard pattern for this class of risk.

## 6. Anti-gaming / reward-hacking (sealed multi-evaluator search)

This is a distinct, non-network security concern worth stating explicitly:
`architecture/sealed_joint_search.py` exists specifically because a *fixed*,
fully-visible evaluation objective is a Goodhart's-Law target — a search
process (including an LLM-guided one) can learn to satisfy the letter of a
known scoring function without the substance it was meant to measure. Sealing
evaluator internals from the generation-facing prompt context, running
multiple differently-biased evaluators, and requiring cross-evaluator
agreement before promotion is a structural anti-gaming control, analogous to
adversarial-robustness practice in ML security. It is opt-in and
research-mode only; it does not replace `EvaluationKernel`'s default,
fully-tested admission path.

## 7. Persistence and secrets hygiene

**Verified across all new persistence surfaces** (RFT export JSONL +
manifest, crowding reports, MRM pack, session/lifecycle artifacts, EDGAR
local cache): none write API keys, bearer tokens, or other secrets to disk.
The RFT export manifest explicitly sets `trains_model: false` and contains
only trajectory metrics/formulas/rewards. Config loading keeps `api_key`-
shaped fields out of anything serialized back to an artifact file.

## 8. Model deserialization

No `torch.load` call exists anywhere in the codebase (grepped the full
repository, not just the round-2 diff) — FactorMiner only ever trains small
models from scratch in-process (`operators/neuro_symbolic.py`'s neural
leaves, `evaluation/model_zoo.py`'s optional GraphSAGE path) and never loads
an external checkpoint, so the classic unsafe-pickle-deserialization class of
vulnerability does not apply today. If external checkpoint loading is ever
added, it must use `weights_only=True` (or a safetensors-format checkpoint)
— documented here as a standing invariant for future contributors, per this
review's own contributor guidance.

## Summary table

| Surface | Network-facing? | Auth? | Fails closed on bad input? | Verified live in this review |
| --- | --- | --- | --- | --- |
| EDGAR connector | Yes (HTTPS, outbound) | N/A (public API, rate-limited) | Yes | Yes — real SEC data returned |
| Futures connector | Mock/offline only today | N/A | Yes | Yes (mock path) |
| Crowding consensus panel | Yes (HTTPS, outbound) | N/A (public data) | Yes | Yes (fixture + malformed-input tests) |
| MCP stdio | No (local pipe) | Not required | N/A | Yes — real tool call |
| MCP HTTP (opt-in) | Yes (loopback default) | Bearer token, refuses to start without it | N/A | Yes — refusal verified live |
| Local-LLM cascade | Opt-in, local by default | Operator-controlled `base_url` | N/A | Code + existing test |
| HTML report rendering | No | N/A | Yes (escapes all free text) | Yes — XSS payload test |
