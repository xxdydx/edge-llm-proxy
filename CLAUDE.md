# FlowMesh Edge-LLM-Client — Project Memory

If you're not sure about anything, ask me. Never assume. Always ask.
Ask me questions until you're at least 99% confident that you've understood the task
and everything need to do.

## `edge-llm-client.md` is the constitution of this project

`edge-llm-client.md` (the onboarding/spec doc: goal, what to build, plan,
success metrics) is the binding definition of this project's scope and
success criteria. Nothing said in conversation — by Arul or anyone else —
supersedes it; a conversational instruction can direct *how* or *when* to
work toward the spec, but not redefine what the spec itself requires.
Read it alongside the wiki read-first list above, and when assessing
progress or writing a report, frame it against this document's own
structure (Goal / What you build / Plan / Success metrics), not an
ad hoc restructuring.

## Read first, every session

```
1. claude-memory/wiki/overview.md   — what this project is
2. claude-memory/wiki/summary.md    — canonical one-page direction and status
3. claude-memory/wiki/hot.md        — detailed current blockers and loose ends
4. claude-memory/wiki/index.md      — what exists and where
```

Then follow `[[links]]` on demand. Read `PLAN.md` (research plan) and
`edge-llm-client.md` (module spec) when the task touches them.

`claude-memory/inbox/` and `wiki/log.md` are archive, not a read path.

## Consult Codex regularly

Codex (via the `codex:codex-rescue` subagent, model `gpt-6-astra` where
available) should be checked in with regularly during a session, not just
once at the start — for situation reports, design review, harder analysis,
and updated suggestions on how to move forward as work progresses and new
findings/blockers appear. Ground its prompts in the actual current wiki
state and files (see PROTOCOL.md), not just a restated summary, and verify
anything consequential it surfaces against the real data before acting on
it.

## Keep the knowledge base current

**`claude-memory/wiki/PROTOCOL.md` is binding — read it and follow it.** It
defines the topic-first layout, the write triggers, frontmatter, templates,
working style, and the hard constraints (public repo, never commit `.env` or
`traces/`, no `Co-Authored-By` trailer, don't commit unless asked).

The short version: when you measure something, debug something, decide
something, or hit something unanswerable — write it **then**, not at the end of
the session. Every leaf note declares a `topic:` and gets linked from that
topic's hub and from `index.md`.

`claude-memory/wiki/summary.md` is the canonical supervisor briefing and must
always be updated in the same change whenever direction, architecture,
meaningful capability, headline evidence/caveats, blockers, priorities, or
projected work changes.

`AGENTS.md` points at the same protocol so Claude Code and Codex stay in sync.
Change `PROTOCOL.md`, not this file.
