# Hosted Pilot Security and Operations

The hosted pilot is a durable, tenant-isolated research-job service. It is not
the local `mcp-serve` command, does not expose arbitrary commands, and never
places orders. Its safe surface is deliberately small: validate data, mine a
bounded factor library, run a dataset-bound benchmark, generate a report, and
verify a receipt.

## Threat model

| Threat | Enforced control | Residual risk / operator duty |
| --- | --- | --- |
| Cross-tenant reads | tenant ID comes from the authenticated principal; DB queries and paths bind it again | test authorization after every new tool |
| Path traversal or symlink escape | relative-path normalization, tenant-root containment, symlink rejection before and after jobs | isolate the worker at OS/container level in production |
| Command injection | typed job schemas and fixed argv; no shell; minimal environment | dependencies still process untrusted data |
| Resource exhaustion | request, queue, active-job, runtime, input, log, total-storage, and daily-compute quotas | add infrastructure CPU/memory/process/network limits |
| Lost/replayed requests | idempotency keys, SQLite WAL durability, job leases and crash recovery | back up DB and storage consistently |
| Token or provider-secret theft | high-entropy hashed tenant tokens; provider keys are forwarded only by an explicit worker allow-list and never serialized | use a secret manager, short-lived provider credentials, and egress policy |
| Audit alteration | tenant-specific append-only hash chain | copy audit exports to immutable external storage |
| Learning-data leakage | explicit purpose consent, fixed metrics/context, revocation deletion, per-tenant means, k-anonymity | repeated releases can permit differencing; review snapshots manually |
| Stale artifacts | tenant retention period, dry-run purge, exact tenant deletion confirmation | schedule and monitor enforcement |

SQLite provides durable pilot coordination on one host, not horizontally
scaled consensus. Do not run multiple machines against a network-mounted SQLite
file.

Request-rate checks and daily compute reservations are serialized in the same
database transactions that admit the corresponding event or job, preventing
parallel submissions from bypassing those application quotas. Retention removes
the oldest 1,000 eligible job directories per invocation; repeat the scheduled
command until its dry run is empty.

## Authentication boundary

Pilot credentials are operator-issued opaque bearer tokens. Each token is
hashed at rest and carries explicit scopes:

- `jobs:submit`, `jobs:read`, `jobs:cancel`
- `artifacts:read`, `usage:read`, `consent:write`

The hosted MCP server revalidates the token against the DB on every tool call
and binds the SDK access-token resource and client ID to the configured MCP
resource and tenant. Non-loopback startup requires an HTTPS resource URL.

For production, put a standards-compliant OAuth authorization server in front
of the resource server, validate the resource audience, publish protected
resource metadata, and request least-privilege scopes. The MCP
[authorization specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
defines these requirements. The Python SDK dependency stays on stable 1.x
(`mcp>=1.27,<2`) while 2.x is not stable.

## Provisioning and service startup

```bash
uv sync --extra mcp

uv run factorminer hosted-pilot init \
  --db /srv/factorminer/control.sqlite3 \
  --storage-root /srv/factorminer/tenants

uv run factorminer hosted-pilot tenant-create partner-01 \
  --display-name "Partner 01" \
  --retention-days 30 \
  --db /srv/factorminer/control.sqlite3 \
  --storage-root /srv/factorminer/tenants

uv run factorminer hosted-pilot credential-create partner-01 \
  --label pilot-client \
  --scope jobs:submit --scope jobs:read --scope jobs:cancel \
  --scope artifacts:read --scope usage:read --scope consent:write \
  --expires-days 30 \
  --db /srv/factorminer/control.sqlite3 \
  --storage-root /srv/factorminer/tenants
```

The credential is printed once. Put it directly into the client's approved
secret store. Provision inputs through the operator command; remote MCP callers
cannot name server-side source paths:

```bash
uv run factorminer hosted-pilot put-input partner-01 reviewed-market.csv \
  --name market.csv \
  --db /srv/factorminer/control.sqlite3 \
  --storage-root /srv/factorminer/tenants
```

Run at least one worker and the stateless JSON MCP resource server under a
process supervisor:

```bash
uv run factorminer hosted-pilot worker \
  --worker-id worker-01 \
  --db /srv/factorminer/control.sqlite3 \
  --storage-root /srv/factorminer/tenants

uv run factorminer hosted-pilot serve \
  --issuer-url https://auth.example/ \
  --resource-url https://pilot.example/mcp \
  --host 127.0.0.1 --port 8766 \
  --db /srv/factorminer/control.sqlite3 \
  --storage-root /srv/factorminer/tenants
```

Terminate TLS at a maintained reverse proxy, restrict request bodies and
connection rates there, and expose only the hosted path. Never route the local
single-operator MCP surface to a tenant-facing listener.

Mining jobs need a configured model provider. By default the worker inherits no
provider secrets. If a pilot requires one, the operator may explicitly add, for
example, `--provider-secret-env GOOGLE_API_KEY`; only the three documented
provider-key names are accepted, and the secret value remains in the worker
environment rather than job parameters, logs, DB rows, or artifacts. Prefer a
short-lived secret injected by the process supervisor.

## Operational controls

Before admitting a partner:

- give each tenant a unique ID and storage namespace;
- reduce default quotas to the agreed workload where possible;
- issue the smallest scope set and shortest useful expiry;
- run cross-tenant and traversal tests in the deployment image;
- configure OS-level CPU, memory, process, disk, and egress limits;
- encrypt disks and backups, and log secret access outside this application.
- name a service owner and incident-response delegate, with current contact and
  escalation details in the deployment's private operations system.

Daily operations:

```bash
uv run factorminer hosted-pilot audit-verify partner-01 DB_AND_STORAGE_OPTIONS
uv run factorminer hosted-pilot retention-purge partner-01 DB_AND_STORAGE_OPTIONS
uv run factorminer hosted-pilot retention-purge partner-01 --apply DB_AND_STORAGE_OPTIONS
uv run factorminer hosted-pilot usage-export partner-01 usage.jsonl DB_AND_STORAGE_OPTIONS
```

Replace `DB_AND_STORAGE_OPTIONS` with the same explicit `--db` and
`--storage-root` values. The first retention command is a dry run. Back up the
SQLite DB and tenant root at the same consistency point, test restoration, and
copy verified audit/usage exports to immutable storage.

On a suspected token leak, revoke that credential, issue a new one, inspect the
tenant audit chain and usage, cancel suspicious jobs, and preserve the DB and
logs. For full erasure, export evidence required by policy first, then require
exact confirmation:

```bash
uv run factorminer hosted-pilot tenant-delete partner-01 \
  --confirm partner-01 DB_AND_STORAGE_OPTIONS
```

Tenant records, credentials, observations, and files are deleted. Only a
pseudonymous deletion tombstone remains for audit continuity.

## Consent-based learning

Learning is off until a tenant explicitly grants one fixed purpose. Only the
allow-listed numeric metrics and coarse contexts are accepted; free text,
formulas, positions, returns, prompts, and tenant names are not accepted.
Revocation immediately deletes that tenant's observations for the purpose.

Snapshots bind each observation to the active consent-policy version, average
each tenant first so a high-volume tenant cannot dominate, then suppress groups
below the threshold. The content-derived snapshot ID versions the aggregate.
The operator command defaults to five distinct tenants:

```bash
uv run factorminer hosted-pilot consensus-snapshot \
  aggregate_workflow_metrics aggregate-workflow.json \
  --min-tenants 5 DB_AND_STORAGE_OPTIONS
```

Every published snapshot needs a human privacy review for small groups,
differencing against earlier snapshots, rare context combinations, and whether
the consent policy version still describes the intended use.
