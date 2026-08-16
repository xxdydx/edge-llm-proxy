# FlowMesh Edge-LLM-Client — Agent Instructions

If you're not sure about anything, ask me. Never assume. Always ask. Ask me
questions until you're at least 99% confident that you've understood the task
and everything need to do.

Applies to Codex and any other agent working in this repo. Claude Code reads
`CLAUDE.md`, which carries the same instructions and points at the same protocol
file.

## Read first, every session

```
1. claude-memory/wiki/overview.md   — what this project is
2. claude-memory/wiki/hot.md        — where it stands right now
3. claude-memory/wiki/index.md      — what exists and where
```

Then follow `[[links]]` on demand. Read `PLAN.md` (research plan) and
`edge-llm-client.md` (module spec) when the task touches them.

`claude-memory/inbox/` and `wiki/log.md` are archive, not a read path.

## Keep the knowledge base current

**`claude-memory/wiki/PROTOCOL.md` is binding — read it and follow it.** It
defines the topic-first layout, the write triggers, frontmatter, templates,
working style, and the hard constraints.

Short version: when you measure something, debug something, decide something, or
hit something unanswerable — write it **then**, not at the end of the session.
Every leaf note declares a `topic:` and gets linked from that topic's hub and
from `index.md`.

`claude-memory/` is a plain directory of Markdown files with YAML frontmatter.
No tooling is required to read or write it.

## Hard constraints

- **This repo is PUBLIC.** `.env` (Lumid, Anthropic, GitHub tokens) and
  `traces/` (real prompts and source contents) are gitignored and must never be
  committed. `claude-memory/` is gitignored too — it may quote traces.
- **Do not commit or push unless explicitly asked.** The default is no.
- **No `Co-Authored-By` trailer** in any commit message.
- Confirm before anything outward-facing or hard to reverse — image pushes,
  starting or stopping remote tasks, anything that leaves the machine.
- Arul does the architectural planning. Propose and explain trade-offs; wait for
  a decision rather than widening scope.

## Repo orientation

| path                               | what it is                                              |
| ---------------------------------- | ------------------------------------------------------- |
| `edgeproxy/`                       | the recording proxy and router (FastAPI + httpx)        |
| `edgeproxy/router.py`              | placement policy — local (vLLM) vs cloud                |
| `edgeproxy/trace/replay.py`        | offline policy evaluation over recorded traces          |
| `bootstrap.sh`                     | box-side setup: deps, model, vLLM serve                 |
| `flowmesh-up.sh`                   | laptop-side driver: submit task, SSH, tunnel, bootstrap |
| `Dockerfile` / `ssh-workflow.yaml` | the dev container and its pinned digest                 |
| `PLAN.md`                          | the research plan                                       |
| `claude-memory/wiki/`              | the knowledge base — see `PROTOCOL.md`                  |
