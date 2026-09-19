# W4 checkpoint certification

This directory contains the single Phase-1 checkpoint certification attempt.
It is deliberately non-generative: Claude Code is pointed at a local
capture-only Anthropic-compatible HTTP stub, never at Qwen, DeepSeek, or the
edgeproxy relays. The experiment tests whether a removed container can resume
the selected captured session and records the result as a certificate or
blocker. No Dataset B row is emitted by this experiment.

Run:

```sh
python experiments/new_datasets/w4_branch/certify_resume.py
```

The script uses the existing native-arm64 smoke image and the same container
launch/setup shape as `w3_capture/run_smoke_capture.py`. It loads the selected
request through `JobStore.read_artifact`; it does not modify A/C JSONL files or
the shared store.
