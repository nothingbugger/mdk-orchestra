# Federated memory sharing — design direction

Status: design sketch, not implementation. No code in-tree yet beyond a
placeholder stub. This document exists so that future work on inter-site
pattern sharing starts from a shared understanding rather than a
re-design.

## 1. Introduction

MDK Fleet's agent layer owns a small but load-bearing memory system:
curated pattern files (`agents/*_memory.md`) that distil what the
Orchestra has learned about a specific miner fleet — which false-positive
shapes to discount, which precursor signatures reliably escalate, which
environmental confounders matter. The curator writes these files
periodically from decision history; specialists read them on every call.

Today this memory is **local to one deployment**. A mining operation
running MDK Fleet on its own telemetry builds up patterns that only its
own agents see. Two observations motivate this document:

1. **Patterns generalise better than raw telemetry does.** A curated
   pattern like *"isolation-forest hashrate flags on miners with
   baseline variance > σ₀ are ~unanimously noise"* is hardware-shaped,
   not site-shaped. Another operator's fleet benefits from it
   immediately, *without* sharing any telemetry.
2. **The natural distribution channel is peer-to-peer.** There is no
   business reason for a central MDK pattern broker: operators don't
   want a third party seeing their pattern library any more than they
   want one seeing their telemetry. And there is no payment /
   reputation economy to build around these patterns — they are a
   public good among operators who opt in.

**Problem statement.** Allow MDK Fleet deployments to publish curated
patterns to, and merge curated patterns from, other consenting
deployments, without: (a) leaking telemetry or operational metadata,
(b) requiring a central broker or any MDK-operated infrastructure,
(c) coupling the agent layer to any specific transport.

**Non-goals.** No payment rails, no reputation scoring economy, no
pattern-licensing scheme, no attempt to detect or punish freeloaders.
Nodes that only consume and never publish are fine — the marginal cost
of a pattern to its author is zero once curated.

## 2. Requirements

| # | Requirement | Why it matters |
|---|---|---|
| R1 | Privacy by construction — a pattern must not contain raw telemetry, miner IDs, site identifiers, or timestamps that could fingerprint a deployment | Patterns are the output of curation; curation already drops specifics. R1 is enforced at publish time, not at transport. |
| R2 | No central broker | No MDK-operated server in the path; deployments exchange directly. |
| R3 | Auditability via signatures | Each published pattern is signed by an origin keypair so recipients can decide whose patterns to trust and build local allow-lists. |
| R4 | Optional pseudonymity | Publishing identity is a keypair, not a legal identity. Operators may rotate keys or publish under multiple keys. |
| R5 | Censure resistance | Transport should survive one-relay takedown and network-level blocks; this is the case for anyone running the agent over restricted networks. |
| R6 | Selective merge | Recipients choose what to merge based on policy (confidence threshold, origin allow-list, domain tag); nothing is auto-merged. |
| R7 | Zero breaking change to the existing agent layer | Federation is additive — removing it leaves MDK Fleet running exactly as before. |

## 3. Pattern data model

A federated pattern is the existing curated pattern, wrapped in a
signed envelope. The payload schema is the same one `agents/curator.py`
already produces; only the outer envelope is new.

```yaml
# Canonical envelope (YAML for readability; on-wire may be JSON)
schema_version: 1
payload:
  pattern_id: "xgboost_hashrate_fp_unanimous_noise"
  domain: "hashrate"          # voltage | hashrate | environment | power | maestro
  confidence: 0.87             # curator's confidence at write time
  sample_size: 41              # how many decisions this was distilled from
  body: |
    # free-form curated pattern text, same format as agents/*_memory.md entries
metadata:
  origin_pubkey: "ed25519:3f2a..."    # R4: pseudonymous
  created_at: "2026-04-22T03:14:00Z"  # envelope creation, NOT decision times
  version: 3                          # monotonic per pattern_id per origin
  domain_tag: "bitcoin-mining/asic/s19-class"  # hardware family, coarse
signature: "ed25519:..."               # over canonical(payload || metadata)
```

**Canonicalisation.** Signing requires a deterministic byte
representation. Use JSON Canonical Form (JCS, RFC 8785) — not
pretty-printed YAML — for the signing input. The YAML above is the
human-readable view.

**What is NOT in the envelope.** No miner IDs, no site IDs, no raw
telemetry windows, no decision IDs. The curator already strips these
at write time; federation inherits that property. R1 is enforced by a
publish-time linter that rejects envelopes whose `body` matches
telemetry-shaped regexes (IP, UUID, `m\d{3}` miner IDs, RFC3339
timestamps inside payload).

**Domain tag.** Coarse hardware family, not a site identifier. Two
operators running S19-class hardware share the same tag; the tag does
not narrow the publisher down.

## 4. Direction 1 — BitTorrent over Tor

**Topology.** Each publisher runs an onion service that exposes a
static directory of signed pattern envelopes plus a magnet-style
torrent index. Subscribers fetch over Tor using a Tor-routed BT client
(e.g. the onion-capable libtorrent configurations documented in
TorBT-style setups). Discovery uses a small set of bootstrap onion
addresses shared out-of-band, plus the Tor DHT.

**Why BT.** Pattern envelopes are tiny (single KB) but a full library
is a few MB; BT's content-addressed, incrementally-fetchable model
keeps transfers cheap and survives a publisher being offline as long as
one seeder is up.

```python
# agents/federated_sync.py — sketch, not implemented
def publish(pattern_file: Path, keypair: Ed25519Keypair) -> None:
    envelope = wrap(pattern_file, keypair)
    lint_no_telemetry(envelope)           # R1
    written = write_to_onion_dir(envelope) # local onion service serves this dir
    update_torrent_index(written)          # regenerate .torrent over the dir

def pull(bootstrap_onions: list[str]) -> list[Envelope]:
    envelopes = []
    for onion in bootstrap_onions:
        envelopes += fetch_torrent_over_tor(onion)
    return [e for e in envelopes if verify_signature(e)]
```

**Costs.** Each publisher runs `tor` + an onion service + a BT client
configured for SOCKS-over-Tor. Operationally heavier than Nostr
(Section 5), but more resilient to relay takedowns because any seeder
keeps the content alive.

## 5. Direction 2 — Nostr over Tor relays

**Topology.** Each pattern envelope is a signed Nostr event (`kind:
30078` parameterised replaceable event, keyed on `pattern_id`).
Publishers post to a handful of relays; subscribers subscribe with a
tag filter. Relays speak WebSocket and are commonly reachable as
`.onion` addresses, so the entire pub/sub path can run over Tor.

**Why Nostr.** Drastically simpler than BT — no DHT, no seeding, no
onion service of one's own. Signed-event model matches R3 and R4
natively. Relay plurality gives R5: any relay can disappear without
loss.

```python
# agents/federated_sync.py — sketch, not implemented
def publish_nostr(envelope: Envelope, keypair: Ed25519Keypair,
                  relays: list[str]) -> None:
    event = {
        "kind": 30078,
        "tags": [
            ["d", envelope.payload.pattern_id],
            ["domain", envelope.payload.domain],
            ["hw", envelope.metadata.domain_tag],
        ],
        "content": canonical_json(envelope),
        "pubkey": keypair.pub_hex,
        "created_at": int(time.time()),
    }
    event["sig"] = keypair.sign(event_id(event))
    for relay in relays:  # relay URLs can be ws+onion
        send_over_tor(relay, event)

def subscribe_nostr(relays: list[str], domain: str) -> Iterator[Envelope]:
    filt = {"kinds": [30078], "#domain": [domain]}
    for event in relay_subscribe_over_tor(relays, filt):
        env = parse_envelope(event["content"])
        if verify_signature(env) and verify_event_sig(event):
            yield env
```

**Costs.** A `tor` daemon and a Nostr client library. No onion
service of one's own required. Trade-off vs BT: if all the relays
one subscribes to drop the event (e.g. retention policy), the pattern
is gone from that subscriber's view until the publisher re-posts.

## 6. Merge policy

The merge step decides whether an incoming envelope becomes an entry
in the local `agents/*_memory.md`. This is the most important piece —
it is where an operator's trust model is expressed. Default posture
is conservative: nothing auto-merges without matching policy.

```yaml
# config/federated_merge.yaml — proposed
min_confidence: 0.75           # payload.confidence threshold
min_sample_size: 20
allowed_domain_tags:
  - bitcoin-mining/asic/s19-class
origin_trust:
  mode: "allowlist"            # allowlist | signed-by-known | open
  pubkeys:
    - "ed25519:3f2a..."        # known collaborator
    - "ed25519:9b1c..."
dedup:
  on_same_pattern_id: "prefer_higher_confidence"  # or "prefer_local"
on_domain_tag_mismatch: "reject"
```

**Dedup.** If `pattern_id` already exists locally (same slug,
different origin or version), the policy chooses: keep local, prefer
higher-confidence incoming, or store both under a namespaced id.
Default: `prefer_local` — the curator's own output is always
privileged over incoming.

**Shadow merge.** Optionally, incoming patterns can land in a
read-only `agents/*_memory.federated.md` that specialists consult as a
secondary source. This isolates federated content from the curator's
loop and makes it trivial to A/B the effect of federation without
contaminating the local library.

## 7. Threat model

**Protects against:**

- *Telemetry leak via pattern content.* R1 linter + curator already
  stripping specifics. A publisher who misconfigures the linter
  still signs what they send, so leakage is attributable, not silent.
- *Centralised surveillance of who-talks-to-whom.* Tor transport
  breaks the network-metadata channel; relays and seeders see only
  Tor circuits.
- *Single point of failure in the distribution channel.* Both
  directions survive one-relay (or one-seeder) takedown.

**Does not protect against:**

- *Pattern inference revealing operations.* A publisher who posts
  *"hashrate drops on fleet after ambient > 42 °C consistently mean
  real failure"* has disclosed that they operate in a warm
  environment. The linter cannot catch this — it is semantic, not
  syntactic. Mitigation: publishers review patterns before enabling
  federation; some patterns stay local.
- *Sybil attacks on open subscribe.* An adversary running many
  pubkeys can flood the namespace. Mitigation: the default merge
  policy is `allowlist`, so open subscribe is opt-in per operator.
- *Replay / stale patterns.* A pattern version that was once correct
  may no longer be. Mitigation: monotonic `version` field per
  `(origin_pubkey, pattern_id)` lets subscribers ignore older
  versions.
- *Pattern poisoning by a trusted origin gone bad.* A compromised
  key on an allow-list can push bad patterns. Mitigation: merges
  are logged; operators can revoke a pubkey and `agents/federated_sync.py`
  should support a "garbage-collect merged-from" pass that removes
  everything previously merged from a revoked key.

## 8. Compatibility

Federation is **strictly additive**. Concretely:

- `agents/curator.py` is untouched. The curator still writes
  `agents/<domain>_memory.md` from local decisions only.
- A new module `agents/federated_sync.py` is added as a placeholder
  stub exposing `publish()` and `subscribe()` that both raise
  `NotImplementedError`. Importing the agent layer does not require
  Tor or any network client.
- `config/federated_merge.yaml` is optional. If absent, merge is
  disabled and federation is a no-op.
- A feature flag `MDK_FEDERATION_ENABLED=0/1` gates
  `federated_sync.subscribe()` being called from a background task.
  Default is `0`.
- Specialist read path is unchanged: they read the same
  `<domain>_memory.md` file. If shadow-merge is used (Section 6),
  they additionally read `<domain>_memory.federated.md` as a lower-
  priority appendix.

Removing federation is a two-step revert: delete the stub and the
config file. Nothing in the Orchestra's decision loop depends on it.

## 9. Roadmap

Four phases, each independently shippable and useful on its own.

**Phase 1 — Manual export/import.** A CLI:
`python -m agents.federated_sync export <pattern_id> --out file.json`
and `... import file.json`. No network. Operators exchange files
out-of-band (email, USB) and exercise the envelope + signature +
linter end-to-end. This is the cheapest way to validate the data
model before committing to a transport.

**Phase 2 — Static Tor hidden service.** Publisher exposes a
read-only directory of signed envelopes over a Tor onion service.
Subscribers `curl --socks5-hostname` the directory and apply merge
policy. No pub/sub, no DHT, no BT complexity. Good enough for a
handful of collaborating operators.

**Phase 3 — BitTorrent vs Nostr, pick one.** Evaluate both
transports against a real small fleet of publishers. Decision
criteria: operational overhead, resilience to takedown, library
maturity in Python. The document deliberately does not pre-commit.

**Phase 4 — P2P trust graph.** Operators sign "I trust pubkey X for
domain Y" statements; merge policy can follow transitive trust up to
a bounded depth. Keeps things decentralised; avoids a single
authority. Only worth building once Phase 3 has enough publishers
for transitive trust to be meaningful.

## 10. References

- Tor Project — *Onion Services Protocol*. Specification at
  `spec.torproject.org`.
- BitTorrent Enhancement Proposals — in particular BEP 3 (core
  protocol) and BEP 5 (DHT). `bittorrent.org/beps`.
- Nostr NIPs — NIP-01 (basic protocol), NIP-33 (parameterised
  replaceable events, now generalised into `kind: 30000–39999`).
  `github.com/nostr-protocol/nips`.
- RFC 8785 — *JSON Canonical Form* (JCS). Used for signing input.
- RFC 8032 — *Edwards-Curve Digital Signature Algorithm (EdDSA)*
  for the Ed25519 keypair.

No tool endorsement is implied. Mentions are structural ("a
libtorrent-class client", "a Nostr relay") rather than product
recommendations; the transport directions are roughly equivalent
designs regardless of which specific implementation is chosen in
Phase 3.
