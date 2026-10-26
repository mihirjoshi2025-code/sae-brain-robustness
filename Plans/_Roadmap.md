---
tags: [plan, roadmap]
---
# Roadmap — 14-week semester, 4 phases, 3 gates

> [!note] Timeline assumption
> Week ranges assume a **14-week semester**; shift to match the actual start date. **A missed gate means narrowing scope, not stopping** (e.g., cut to one dataset).

## The arc
```
Phase 1 · Setup        weeks 1–3   ──▶ Gate A: dense baseline reproduced
Phase 2 · Core grid    weeks 4–8   ──▶ Gate B: controls on all 12 configs; taxonomy κ ≥ 0.7
Phase 3 · Analysis     weeks 9–11  ──▶ Gate C: every claim scored; noise ceilings reported
Phase 4 · Write-up     weeks 12–14 ──▶ Results grid, public repo, club presentation
```

## Phase files (week-by-week)
- [[Phase 1 - Setup (Weeks 1-3)]]
- [[Phase 2 - Core Grid (Weeks 4-8)]]
- [[Phase 3 - Analysis (Weeks 9-11)]]
- [[Phase 4 - Write-up (Weeks 12-14)]]

## The three gates (go/no-go)
| Gate | When | Pass condition | If missed |
|---|---|---|---|
| **A** | end of Week 3 | Dense-feature baseline reproduced on one dataset | Cut to **one dataset** before expanding |
| **B** | end of Week 8 | Controls run on **all 12 configs**; taxonomy agreement **κ ≥ 0.7** | Narrow scope (fewer models/sizes); re-do labeling scheme |
| **C** | end of Week 11 | **Every claim scored**; noise ceilings reported for each | Drop the stretch goal; focus write-up on what's done |

## Critical path & parallelism
The **[[Roles and Workstreams|encoding pipeline]]** is the critical path. Until it's ready, the other three workstreams build against a small **cached test fixture**. See [[Roles and Workstreams]].

## Standing weekly rituals (every week)
- **Monday sync** (lead-run): blockers, gate progress, who's waiting on whom.
- **arXiv scan**: check for new brain↔LLM/SAE papers; adjust claims if scooped.
- **Results table hygiene**: every run logged with config + seed + score + noise ceiling.
- **Decision log**: record any config/scope change and why.

## Pre-kickoff checklist (Week 0)
- [ ] Everyone reads the first three papers (see [[_Reading List]]).
- [ ] Create the shared GitHub repo: `pipeline/ features/ taxonomy/ controls/ configs/`.
- [ ] Decide **compute source** (club cluster / dept / cloud).
- [ ] Shortlist models by **SAE layer/size coverage** (don't finalize the model list until checked).
- [ ] Assign the four workstream owners.
