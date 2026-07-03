# Efficacy Task Corpus — YAML Schema

This directory holds the **machine-checkable, hand-verified task set** with ground
truth for the Reflex AI-coding efficacy experiment ([REF-176](/REF/issues/REF-176) /
[REF-178](/REF/issues/REF-178)). Each `*.yaml` file groups the tasks for one pinned
repository from the corpus (see [`../repos.md`](../repos.md)).

Ground truth is the **credibility anchor** of the whole experiment: if it is wrong,
every downstream metric is wrong. Two rules keep it honest:

1. **Non-circular.** Ground truth is derived by tools **independent of Reflex**
   (ripgrep + manual inspection + tree-sitter), never by Reflex itself. Otherwise we
   would be grading Reflex against its own output.
2. **Regenerable, not transcribed.** `match_set` / `file_set` answer keys are stored
   as an *oracle spec* (a ripgrep query) plus a `count` + `sha256` of the result.
   [`validate.py`](../validate.py) re-runs the oracle against the pinned checkout and
   checks the count and checksum, so the answer key can never silently drift from the
   code.

## File layout

```
benches/efficacy/
  oracle.py          # canonical ground-truth normalization (shared by authoring + validate)
  validate.py        # parses every task, resolves every ground-truth ref in the pinned repo
  repos.md           # pinned repo SHAs + fetch instructions
  corpus/            # (gitignored) checkouts of the pinned repos, created by fetch
  tasks/
    SCHEMA.md        # this file
    reflex.yaml      # tasks against the Reflex repo
    ripgrep.yaml     # tasks against BurntSushi/ripgrep
    tokio.yaml       # tasks against tokio-rs/tokio
```

## Top-level file structure

Each task file is a mapping with a single `tasks:` list:

```yaml
tasks:
  - id: reflex-findall-extract_symbols
    repo: reflex
    repo_sha: d2935f48f5abea2a76b479040a23478155be9bb0
    category: find_all_usages
    expect_reflex: win
    prompt: "Find every occurrence of the identifier `extract_symbols` ..."
    ground_truth:
      type: match_set
      oracle: { pattern: 'extract_symbols', path: 'src', flags: ['-w'] }
      expected_count: 122
      expected_sha256: '91ad6e25a34481fd832eb4129128889bda187529a00917661af8de7ed0e6a1e4'
    oracle_notes: "Cross-checked by (1) ... (2) ..."
```

## Task fields

| Field          | Type   | Required | Notes |
|----------------|--------|----------|-------|
| `id`           | string | yes | Globally unique, kebab-case: `<repo>-<category>-<target>`. |
| `repo`         | string | yes | Repo id; MUST appear in `repos.md` (`reflex` \| `ripgrep` \| `tokio`). |
| `repo_sha`     | string | yes | 40-char pinned commit SHA; MUST match the SHA for `repo` in `repos.md`. |
| `category`     | enum   | yes | One of the categories below. |
| `expect_reflex`| enum   | yes | Pre-declared expectation: `win` \| `neutral` \| `lose`. Guards against cherry-picking (see acceptance criterion). Set **before** any run. |
| `prompt`       | string | yes | The natural-language task handed to the agent. Repo-specific and self-contained. |
| `ground_truth` | map    | yes | Typed; see below. |
| `oracle_notes` | string | yes | How the ground truth was cross-checked by ≥2 independent tools; name the exact commands/methods. |

### Categories

Mapped from the [REF-176 plan](/REF/issues/REF-176#document-plan):

| Category              | Plan # | Ground-truth `type` |
|-----------------------|--------|---------------------|
| `locate_definition`   | 1 | `location` |
| `find_all_usages`     | 2 | `match_set` |
| `refactor_scope`      | 3 | `file_set` |
| `dependency_imports`  | 4 | `match_set` (scoped to one file) |
| `reverse_dependency`  | 4 | `file_set` |
| `hotspot`             | 5 | `ranking` |
| `cross_module`        | 6 | `match_set` (+ dispatch note) |
| `comprehension`       | 7 | `rubric` |
| `negative_control`    | 8 | `match_set` (literal `-F`) |

> **Note on cross-language (category 6).** The CEO-locked corpus is all-Rust
> (Reflex + ripgrep + tokio), so the "cross-language, where is this API implemented
> and consumed" category is realised as **cross-module** within a repo (e.g. the 15
> per-language `extract_symbols` implementations dispatched from `parsers/mod.rs`). A
> true multi-language task needs a polyglot repo; that is tracked as a possible corpus
> expansion, not blocked here.

## Ground-truth types

### `location` — a single definition site
```yaml
ground_truth:
  type: location
  file: src/trigram.rs
  line: 184
  expect_regex: '^pub struct TrigramIndex \{'   # Python re, matched against the line's text
```
**Validator:** the file exists at `repo_sha`, has ≥ `line` lines, and line `line`
matches `expect_regex`.

### `match_set` — the complete set of occurrences (find-all / imports / control)
```yaml
ground_truth:
  type: match_set
  oracle: { pattern: 'extract_symbols', path: 'src', flags: ['-w'] }
  expected_count: 122
  expected_sha256: '91ad6e25...'
```
`oracle.pattern` is a ripgrep regex (or literal when `flags` includes `-F`).
`oracle.flags` are extra rg flags (`-w` word boundary, `-F` fixed string, `-i` case
-insensitive). **Validator:** re-runs the oracle (`oracle.py match_set`) against the
checkout and asserts `count` and `sha256`.

### `file_set` — the set of files (refactor scoping / reverse deps)
```yaml
ground_truth:
  type: file_set
  oracle: { pattern: 'QueryEngine', path: 'src', flags: ['-w'] }
  expected_count: 13
  expected_sha256: 'a0ea43d7...'
  expected_files: [src/cli/query.rs, src/lib.rs, ...]   # optional, for human audit
```
**Validator:** re-runs `oracle.py file_set`; asserts `count` + `sha256`; if
`expected_files` is present, asserts it equals the regenerated set exactly.

### `ranking` — ordered-by-metric answer (hotspots)
```yaml
ground_truth:
  type: ranking
  metric: file_set_count           # count of files matching the per-candidate pattern
  pattern_template: 'crate::{name}\b'
  path: src
  flags: []
  candidates: [models, cache, parsers, indexer, content_store, query]
  expected_top: models
  expected_counts: { models: 46, cache: 34, parsers: 19 }   # optional audit values
```
**Validator:** for each `candidate`, substitutes it into `pattern_template` and
computes the `file_set` count; asserts the argmax equals `expected_top`. Any
`expected_counts` entries are asserted exactly.

### `rubric` — comprehension (graded later by a blinded LLM judge)
```yaml
ground_truth:
  type: rubric
  expected_files: [src/indexer.rs]           # files the correct answer must reference
  required_facts: ["blake3 content hashing", "skip reindex when hash unchanged"]
```
**Validator:** asserts every `expected_files` entry exists at `repo_sha`.
`required_facts` are **not** machine-checked here — they feed the Phase 5 answer-quality
rubric. Comprehension ground truth is intentionally the softest; keep these few.

## Determinism guarantees

- All ripgrep oracles run with fixed flags (`--no-heading --line-number
  --with-filename --color never`) inside `oracle.py`, respecting the repo's
  `.gitignore` (the same file set Reflex indexes).
- `sha256` is over the sorted-unique `path:line` (or `path`) lines joined with `\n`.
- Pinned SHAs make the corpus content immutable across replicate trials.
