"""Build the Phase 1 workload from MBPP+ v0.2.0.

Loading path (fidelity first):
  * ``evalplus.data.get_mbpp_plus`` returns the pinned dataset with the
    official per-task input deserialisation applied.
  * for each task, contract-filter the base+plus inputs (drop inputs whose
    precondition ``contract`` raises), then run the TRUSTED canonical solution
    on each surviving input via ``evalplus.gen.util.trusted_exec`` to get the
    aligned ``expected`` outputs and per-input reference times.
  * inputs and expected outputs are ``serde``-encoded (tuple/set-preserving)
    so the Docker grader reconstructs exact Python objects.

Only the canonical (trusted repo) solution runs on the host here; no model
code. The EvalPlus content hash and the local gz SHA256 are recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from random import Random

from .config import Phase1Config
from .schema import DatasetManifest, Problem
from .serde import encode

# EvalPlus MBPP oracles where the reference output is coerced to "not None".
# Their canonicals can return unjson-able objects (e.g. re.Match); the label
# is only None-vs-not-None, matching evalplus trusted_exec(output_not_none=True)
# and untrusted_check's MBPP_OUTPUT_NOT_NONE_TASKS handling.
_OUTPUT_NOT_NONE = {"check_str", "text_match_three", "text_starta_endb"}


def _group_of(task_id: str, n_groups: int) -> int:
    """Stable hash bucket for ``task_id``. This guarantees no single task's
    evidence appears in two splits; it does NOT cluster near-duplicate MBPP
    problems (that is not attempted here)."""
    return int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % n_groups


def _param_names(canonical_solution: str, entry_point: str) -> list[str]:
    import ast

    try:
        tree = ast.parse(canonical_solution)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return [a.arg for a in node.args.args]
    return []


def _run_contract(entry_point: str, contract: str, inp, names: list[str] | None = None) -> bool:
    """True if this input satisfies the task's precondition contract. EvalPlus
    contracts are ``assert`` lines referencing the function's parameter names;
    we bind the positional inputs to those names and exec them."""
    if not contract or not contract.strip():
        return True
    names = names or []
    args = list(inp) if isinstance(inp, (list, tuple)) else [inp]
    ns: dict = {names[i] if i < len(names) else f"arg{i}": v for i, v in enumerate(args)}
    body = "\n".join(
        ln.strip() for ln in contract.splitlines() if ln.strip().startswith("assert")
    )
    if not body.strip():
        return True
    try:
        exec(body, {}, ns)
        return True
    except Exception:
        return False


def build_dataset(cfg: Phase1Config) -> tuple[list[Problem], DatasetManifest, dict]:
    from evalplus.data.mbpp import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.gen.util import trusted_exec

    ds = get_mbpp_plus(mini=False, noextreme=False, version=cfg.dataset_version)
    evalplus_hash = get_mbpp_plus_hash(version=cfg.dataset_version)
    gz_sha = (
        hashlib.sha256(cfg.dataset_gz.read_bytes()).hexdigest()
        if cfg.dataset_gz.is_file()
        else None
    )

    items = sorted(ds.items(), key=lambda kv: int(kv[0].split("/")[1]))
    rng = Random(cfg.seed)
    order_ids = [tid for tid, _ in items]
    rng.shuffle(order_ids)

    # Walk the shuffled order and keep the first n_tasks that yield a
    # gradeable problem (some MBPP+ tasks have contracts that reject all
    # their inputs, or a canonical that raises). This hits the exact target
    # without changing the deterministic selection order.
    n_win = cfg.n_windows
    per_win = max(1, (cfg.n_tasks + n_win - 1) // n_win)

    problems: list[Problem] = []
    dropped_no_inputs = dropped_canonical = 0
    for tid in order_ids:
        if len(problems) >= cfg.n_tasks:
            break
        t = ds[tid]
        i = len(problems)
        ep = t["entry_point"]
        names = _param_names(t["canonical_solution"], ep)
        base = [x for x in (t.get("base_input") or []) if _run_contract(ep, t.get("contract", ""), x, names)]
        plus = [x for x in (t.get("plus_input") or []) if _run_contract(ep, t.get("contract", ""), x, names)]
        if not base and not plus:
            dropped_no_inputs += 1
            continue
        try:
            exp_base, rt_base = trusted_exec(t["canonical_solution"], base, ep, record_time=True, output_not_none=(ep in _OUTPUT_NOT_NONE)) if base else ([], [])
            exp_plus, rt_plus = trusted_exec(t["canonical_solution"], plus, ep, record_time=True, output_not_none=(ep in _OUTPUT_NOT_NONE)) if plus else ([], [])
        except Exception:
            dropped_canonical += 1
            continue
        prob = Problem(
            task_id=tid,
            entry_point=ep,
            prompt=t["prompt"],
            canonical_solution=t["canonical_solution"],
            base_input=[encode(x) for x in base],
            plus_input=[encode(x) for x in plus],
            expected=[encode(x) for x in list(exp_base) + list(exp_plus)],
            ref_time=[float(x) for x in list(rt_base) + list(rt_plus)],
            atol=float(t.get("atol") or 0.0),
            contract=t.get("contract", ""),
            n_base=len(base),
            n_plus=len(plus),
            task_group=_group_of(tid, cfg.n_task_groups),
            window=min(n_win - 1, i // per_win),
            order=i,
        )
        try:
            json.dumps(asdict(prob))  # reject any task whose expected outputs aren't serialisable
        except TypeError:
            dropped_canonical += 1
            continue
        problems.append(prob)

    gc: dict[str, int] = {}
    wc: dict[str, int] = {}
    for p in problems:
        gc[str(p.task_group)] = gc.get(str(p.task_group), 0) + 1
        wc[str(p.window)] = wc.get(str(p.window), 0) + 1
    manifest = DatasetManifest(
        version=cfg.dataset_version,
        seed=cfg.seed,
        selected=len(problems),
        available=len(items),
        task_group_counts=dict(sorted(gc.items(), key=lambda kv: int(kv[0]))),
        window_counts=dict(sorted(wc.items(), key=lambda kv: int(kv[0]))),
    )
    provenance = {
        "evalplus_mbpp_plus_hash": evalplus_hash,
        "local_gz_sha256": gz_sha,
        "local_gz_path": str(cfg.dataset_gz),
        "dropped_no_contract_passing_inputs": dropped_no_inputs,
        "dropped_canonical_exec_failed": dropped_canonical,
        "target_n_tasks": cfg.n_tasks,
        "note": "walked the seeded shuffle order, kept the first target_n_tasks gradeable tasks",
    }
    return problems, manifest, provenance


def build_one(task_id: str, cfg: Phase1Config) -> Problem:
    """Build a single Problem (no shuffle / loop). For smoke tests and unit
    tests -- same construction path as ``build_dataset``."""
    from evalplus.data.mbpp import get_mbpp_plus
    from evalplus.gen.util import trusted_exec

    t = get_mbpp_plus(version=cfg.dataset_version)[task_id]
    ep = t["entry_point"]
    names = _param_names(t["canonical_solution"], ep)
    base = [x for x in (t.get("base_input") or []) if _run_contract(ep, t.get("contract", ""), x, names)]
    plus = [x for x in (t.get("plus_input") or []) if _run_contract(ep, t.get("contract", ""), x, names)]
    exp_base, rt_base = trusted_exec(t["canonical_solution"], base, ep, record_time=True, output_not_none=(ep in _OUTPUT_NOT_NONE)) if base else ([], [])
    exp_plus, rt_plus = trusted_exec(t["canonical_solution"], plus, ep, record_time=True, output_not_none=(ep in _OUTPUT_NOT_NONE)) if plus else ([], [])
    return Problem(
        task_id=task_id, entry_point=ep, prompt=t["prompt"],
        canonical_solution=t["canonical_solution"],
        base_input=[encode(x) for x in base], plus_input=[encode(x) for x in plus],
        expected=[encode(x) for x in list(exp_base) + list(exp_plus)],
        ref_time=[float(x) for x in list(rt_base) + list(rt_plus)],
        atol=float(t.get("atol") or 0.0), contract=t.get("contract", ""),
        n_base=len(base), n_plus=len(plus),
        task_group=_group_of(task_id, cfg.n_task_groups), window=0, order=0,
    )


def save_dataset(problems, manifest, provenance, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "problems.jsonl").open("w") as fh:
        for p in problems:
            fh.write(json.dumps(asdict(p)) + "\n")
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps({**asdict(manifest), "provenance": provenance}, indent=2) + "\n"
    )


def load_dataset(out_dir: Path) -> list[Problem]:
    path = out_dir / "problems.jsonl"
    if not path.is_file():
        raise SystemExit(f"no problems.jsonl under {out_dir}")
    return [Problem(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def split_by_group(problems: list[Problem], cfg: Phase1Config):
    groups = sorted({p.task_group for p in problems})
    Random(cfg.seed + 1).shuffle(groups)
    n = len(groups)
    n_tr = max(1, round(n * cfg.train_frac))
    n_va = max(1, round(n * cfg.val_frac))
    tr, va, te = set(groups[:n_tr]), set(groups[n_tr : n_tr + n_va]), set(groups[n_tr + n_va :])
    return (
        [p for p in problems if p.task_group in tr],
        [p for p in problems if p.task_group in va],
        [p for p in problems if p.task_group in te],
    )
