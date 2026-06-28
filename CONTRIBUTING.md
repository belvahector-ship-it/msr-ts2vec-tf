# Contributing — Branch Strategy and Development Workflow

This document defines the Git workflow for the project. It exists to protect
experimental reproducibility: a merged research module must be stable before
any downstream module is built on top of it.

---

## Branch Structure

```
main
 └── develop
      ├── feat/M0-bootstrap
      ├── feat/M1-acquisition
      ├── feat/M2-validation
      ├── feat/M3-alignment
      ├── feat/M4-features
      ├── feat/M5-split
      ├── feat/M6-windows
      ├── feat/M7-ts2vec-wrapper
      ├── feat/M8-training
      ├── feat/M9-fusion
      ├── feat/M10-clustering
      ├── feat/M11-evaluation
      ├── feat/M12-visualization
      ├── feat/M13-runner
      ├── feat/M14-statistics
      ├── feat/M15-artifacts
      └── fix/<short-description>
```

### Branch roles

| Branch | Purpose | Merge target |
|--------|---------|--------------|
| `main` | Stable, publication-ready state. Tagged at each milestone. | — |
| `develop` | Integration branch. All feature branches merge here first. | `main` (at milestones) |
| `feat/M*` | One branch per IMP-01 module. Deleted after merge. | `develop` |
| `fix/*` | Bug fixes on completed modules. | `develop` |

---

## Workflow

### Starting a new module

```bash
git checkout develop
git pull origin develop
git checkout -b feat/M1-acquisition
```

### During development

Commit frequently. Commit messages use the format:

```
<type>(<scope>): <short description>

[optional body]
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`

Examples:
```
feat(M1): implement BinanceDownloader with retry logic
test(M1): add V-DATA-001 row count validation test
fix(M2): handle duplicate timestamp edge case in validator
docs(M0): update paths.py docstring for branch_checkpoint_dir
```

### Merging a completed module

A module may only merge to `develop` when its IMP-01 Definition of Done
is fully satisfied:

1. All DoD checklist items are complete
2. All mapped DS-04 validation tests pass
3. No linting errors (`mypy src/`, `pytest tests/`)
4. Module-level docstring is complete

```bash
git checkout develop
git merge --no-ff feat/M1-acquisition -m "merge(M1): data acquisition complete — DoD satisfied"
git branch -d feat/M1-acquisition
```

### Milestone tags

At each IMP-01 milestone, `develop` is merged to `main` and tagged:

```bash
git checkout main
git merge --no-ff develop -m "milestone: M2 data pipeline complete"
git tag -a v0.1.0-data-pipeline -m "Milestone 2: data pipeline complete"
git push origin main --tags
```

Planned tags:

| Tag | Milestone |
|-----|-----------|
| `v0.1.0-bootstrap` | Milestone 1 — repository ready |
| `v0.2.0-data-pipeline` | Milestone 2 — data pipeline complete |
| `v0.3.0-representation` | Milestone 3 — representation learning complete |
| `v0.4.0-experiment` | Milestone 4 — experiment complete |
| `v1.0.0-paper-ready` | Milestone 5 — paper-ready outputs complete |

---

## Research Protocol Rules

These rules protect scientific integrity and override convenience:

1. **No research decisions change in code.** ADRs are frozen. If an implementation
   reveals a genuine impossibility, open a formal deviation request — do not
   silently modify behavior.

2. **No module bypasses another module's output.** If M3 produces
   `btc_aligned_1h.parquet`, M4 must read exactly that file. No inline
   re-computation of upstream stages.

3. **No experiment runs until the leakage gate passes.** All V-LEAK tests
   (DS-04 §3.2) must pass before M8 (branch training) begins.

4. **Checkpoint format is frozen at ADR-010.** Do not add or remove keys
   from the checkpoint bundle without a new ADR.

5. **No print() in source modules.** Use `get_logger(__name__)` from
   `src.utils.logging_utils`.
