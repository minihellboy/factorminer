# Design-Partner Pilot Protocol

A design-partner result is credible only when a real external reviewer controls
the input, observes the protocol, and chooses which bounded claims to sign.
The project cannot self-issue that acknowledgment.

## Roles and evidence boundary

| Role | Controls | May assert |
| --- | --- | --- |
| Producer | code, configuration, portable artifacts | what the software ran |
| Partner | proprietary dataset, review key, publication consent | only actions personally observed |
| Verifier | receipt, request, acknowledgment, disclosed verification key | integrity and exact claim binding |

The private receipt stores an HMAC-SHA-256 dataset commitment rather than the
dataset or key. Use distinct random keys for the dataset commitment and the
review acknowledgment. Keep both outside Git, shell history, reports, and
portable artifacts.

## 1. Run the partner-owned dataset

The partner runs the command in its environment or supplies the commitment key
through an approved ephemeral secret file:

```bash
uv run python scripts/run_phase2_benchmark.py \
  --data PARTNER_MARKET_DATA.csv \
  --data-license-class proprietary_licensed \
  --evidence-tier private_partner_observed \
  --commitment-key-file PARTNER_DATASET_KEY.hex \
  --portable-release \
  --runs 10 \
  --seed 42 \
  --n-factors 40 \
  --full-ablation \
  --output output/partner-pilot
```

No raw partner data is copied into the release.

## 2. Prepare a bounded review request

```bash
uv run factorminer partner-review prepare \
  output/partner-pilot/releases/RELEASE_ID \
  output/partner-pilot/review-request.json \
  --partner-pseudonym partner-01 \
  --assertion protocol_observed \
  --assertion dataset_commitment_verified \
  --assertion artifacts_reviewed \
  --assertion limitations_acknowledged \
  --assertion results_reproduced
```

Request only claims the partner is in a position to check. A request is not an
acknowledgment and never changes the receipt's status.

## 3. Partner records the outcome

The feedback file is deliberately non-narrative to reduce accidental disclosure:

```json
{
  "setup_minutes": 30,
  "campaign_completed": true,
  "completion_rate": 1.0,
  "trust_rating": 4,
  "decision_outcome": "pilot_more",
  "adoption_blockers": ["data_integration", "security_review"],
  "repeated_use_intent": "yes",
  "useful_outputs": ["evidence_report", "receipt"],
  "failure_modes": ["none"],
  "requested_integrations": ["market_data", "sso"]
}
```

Allowed decision values are `adopt`, `pilot_more`, `not_now`, `reject`, and
`undisclosed`. Blockers are drawn from a fixed taxonomy. Names, free text,
positions, returns, formulas, and proprietary metadata are rejected.

The partner then selects only the assertions actually observed:

```bash
uv run factorminer partner-review acknowledge \
  output/partner-pilot/review-request.json \
  output/partner-pilot/acknowledgment.json \
  --reviewer-pseudonym reviewer-01 \
  --assertion protocol_observed \
  --assertion artifacts_reviewed \
  --assertion limitations_acknowledged \
  --publication-consent anonymous \
  --key-file PARTNER_REVIEW_KEY.hex \
  --feedback-json PARTNER_FEEDBACK.json
```

`private` consent keeps the acknowledgment internal, `anonymous` permits the
pseudonymous record, and `public` permits the stated reviewer identity. Consent
does not broaden the signed assertions.

The built-in HMAC proves integrity to parties that share the partner-controlled
review key; it is not public-key identity proof because any key holder could
produce a signature. A public named endorsement therefore also requires the
partner's normal approved publication channel or an independently managed
public-key signature. Do not turn `publication_consent: public` into a stronger
identity claim by itself.

## 4. Verify and report honestly

```bash
uv run factorminer partner-review verify \
  output/partner-pilot/review-request.json \
  output/partner-pilot/acknowledgment.json \
  output/partner-pilot/releases/RELEASE_ID \
  --key-file PARTNER_REVIEW_KEY.hex
```

Verification binds the acknowledgment to the exact review request, receipt
release ID, receipt bytes, assertion subset, and HMAC signature. Any edit
invalidates the content-derived ID or signature. Publish only at the consented
level, and report partial completion or rejection as evidence rather than
silently dropping an unfavorable pilot.
