# Efficacy Corpus — Pinned Repositories

The efficacy experiment ([REF-176](/REF/issues/REF-176)) runs every task against a
repository **pinned at a fixed commit SHA** so ground truth is stable across all
replicate trials and all arms. This file is the authoritative pin list and the fetch
recipe. Every `repo_sha` in `tasks/*.yaml` MUST match a SHA below.

> **Corpus scope (CEO-locked).** The [REF-176 plan](/REF/issues/REF-176#document-plan)
> locks the corpus to three Rust repositories: the Reflex repo (thin-slice backbone),
> plus `BurntSushi/ripgrep` and `tokio-rs/tokio` for the full run. This supersedes the
> original REF-178 "4–6 polyglot repos" wording. A polyglot/large-monorepo expansion
> (for a genuine cross-*language* category) remains a possible follow-up but is not in
> the locked scope.

## Pinned repositories

| id        | Repository              | Pin (tag)      | Commit SHA                                 | Size / character |
|-----------|-------------------------|----------------|--------------------------------------------|------------------|
| `reflex`  | this repository         | `main`@bump    | `d2935f48f5abea2a76b479040a23478155be9bb0` | Rust, mid-size; the codebase we own — ground truth manually verifiable. Thin-slice repo. |
| `ripgrep` | `BurntSushi/ripgrep`    | `14.1.1`       | `4649aa9700619f94cf9c66876e9549d83420e16c` | Rust, small/medium, well-understood multi-crate workspace; trigram-friendly. |
| `tokio`   | `tokio-rs/tokio`        | `tokio-1.46.1` | `ab3ff69cf2258a8c696b2dca89a2cef4ff114c1c` | Rust, medium, async; non-trivial symbol graph — stresses dependency/symbol resolution. |

### Notes on the `reflex` pin

- The pinned SHA `d2935f4` is the tip of `main` (`chore: bump version`). The efficacy
  branch (`feature/mcp-efficacy-tests`) adds only files under `benches/efficacy/` and
  `.context/` on top of it, so **`src/`, `tests/`, `Cargo.toml`, and `Cargo.lock` are
  byte-identical** between `main@d2935f4` and the working branch. Ground truth authored
  against the working tree is therefore valid for the pinned SHA. Verified with:
  `git diff --stat d2935f48f5abea2a76b479040a23478155be9bb0 HEAD -- src tests Cargo.toml Cargo.lock` (empty).

## Fetch instructions

External repos are cloned on demand into `benches/efficacy/corpus/<id>` (gitignored —
never committed). The validator ([`validate.py`](./validate.py)) fetches automatically,
but you can prepare them manually:

```bash
cd benches/efficacy
mkdir -p corpus

# ripgrep @ 14.1.1
git clone --filter=blob:none https://github.com/BurntSushi/ripgrep.git corpus/ripgrep
git -C corpus/ripgrep checkout 4649aa9700619f94cf9c66876e9549d83420e16c

# tokio @ 1.46.1
git clone --filter=blob:none https://github.com/tokio-rs/tokio.git corpus/tokio
git -C corpus/tokio checkout ab3ff69cf2258a8c696b2dca89a2cef4ff114c1c
```

For the `reflex` repo, the corpus checkout is this repository itself at `d2935f4`; the
validator resolves it via a `git worktree` at the pinned SHA (falling back to the
current checkout when `src/` is confirmed identical). See `validate.py --help`.

### Verifying a pin

```bash
git -C corpus/ripgrep rev-parse HEAD   # -> 4649aa9700619f94cf9c66876e9549d83420e16c
git -C corpus/tokio   rev-parse HEAD   # -> ab3ff69cf2258a8c696b2dca89a2cef4ff114c1c
```

## Why these SHAs

- **Tagged releases**, not arbitrary commits — reproducible and easy to audit.
- ripgrep `14.1.1` and tokio `1.46.1` are the latest stable releases at corpus-freeze
  time (2026-07-02), so the code is representative of what an agent would search today.
- Annotated-tag SHAs are the **dereferenced commit** (`^{}`) objects, i.e. the actual
  tree a `checkout` lands on.
