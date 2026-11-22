# AI handoff: start here

This folder lets any AI assistant (or new club member) pick up the SFU Racerbot
coordination work without prior context. It was exported from the team coordinator's
notes on 2026-11-22.

## Read in this order

1. [team-rules.md](team-rules.md) — the owner's standing rules. Follow every *high*
   confidence rule; verify *low* ones before relying on them.
2. [project-state.md](project-state.md) — open PRs, work held for review, and pushed
   checkpoint commits.
3. [evidence-log.md](evidence-log.md) — dated decisions and what each experiment did
   and did not show.
4. [Turn-response handoff](turn-response.md) — Car 2's turning rate improved, but its lap time got worse. Keep both defaults.

Then read the repository's own [CLAUDE.md](../../CLAUDE.md), [AGENTS.md](../../AGENTS.md)
and [docs/architecture.md](../architecture.md) before touching driving code.

## The rules that matter most

- Work on an isolated branch and open a pull request; never merge, deploy or run the
  physical car without the owner's explicit go-ahead.
- Commits and PRs are signed as SFU Racerbot only: check author, committer and message
  trailers before every push.
- Never commit run data (bags, recordings, intake inventories) or secrets.
- Never weaken safety: keep the LB deadman, braking margins and test bounds as they are.

## What was left out

- Raw run data and recording inventories from the coordinator's notes.
- Links to evidence files in the coordinator's private workspace (shown as "(private link)");
  the summaries and numbers are kept.
- Internal session identifiers.

These notes are a snapshot. PR states, branches and numbers may have changed: check GitHub
before acting on any of them.
