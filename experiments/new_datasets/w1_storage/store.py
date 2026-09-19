"""Single-coordinator SQLite event log, leases, artifacts, and atomic exports."""
from __future__ import annotations
import gzip, hashlib, json, os, sqlite3, time
from pathlib import Path
from typing import Literal
from .schemas import DATASETS, RECORD_TYPES, RecordBase, ValidationError

class ArtifactMismatch(ValueError): pass

class JobStore:
    """Local store. A SQLite file must not be shared by multiple hosts/NFS writers."""
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "store.sqlite3", timeout=30, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, owner TEXT, lease_until REAL, heartbeat REAL);
        CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, dataset TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(dataset,payload));
        CREATE TABLE IF NOT EXISTS artifacts (sha256 TEXT PRIMARY KEY, length INTEGER NOT NULL, path TEXT NOT NULL);
        """)
        self.journal = self.root / "events.journal.jsonl"; self.artifacts = self.root / "artifacts"; self.artifacts.mkdir(exist_ok=True)

    def claim(self, job_id: str, owner: str, lease_seconds: float = 300) -> None:
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT owner,lease_until FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row and row[1] and row[1] > now: raise RuntimeError(f"job already leased by {row[0]}")
            self.db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?)", (job_id, owner, now+lease_seconds, now)); self.db.execute("COMMIT")
        except Exception: self.db.execute("ROLLBACK"); raise

    def heartbeat(self, job_id: str, owner: str, lease_seconds: float = 300) -> None:
        cur = self.db.execute("UPDATE jobs SET lease_until=?,heartbeat=? WHERE job_id=? AND owner=? AND lease_until>?", (time.time()+lease_seconds,time.time(),job_id,owner,time.time()))
        if cur.rowcount != 1: raise RuntimeError("no active lease owned by caller")

    def release(self, job_id: str, owner: str) -> None:
        cur = self.db.execute("DELETE FROM jobs WHERE job_id=? AND owner=?", (job_id,owner))
        if cur.rowcount != 1: raise RuntimeError("no lease owned by caller")

    def put_artifact(self, raw: bytes) -> dict[str, object]:
        if not isinstance(raw, bytes): raise TypeError("raw artifact must be bytes")
        digest = hashlib.sha256(raw).hexdigest(); path = self.artifacts / f"{digest}.gz"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wb") as fh: fh.write(raw); fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, path)
        self.db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?)", (digest,len(raw),str(path)))
        return {"ref": f"artifact:{digest}", "sha256": digest, "length": len(raw)}

    def read_artifact(self, ref: str) -> bytes:
        if not ref.startswith("artifact:"): raise ArtifactMismatch("invalid artifact reference")
        digest = ref[9:]; row = self.db.execute("SELECT path,length FROM artifacts WHERE sha256=?", (digest,)).fetchone()
        if not row: raise ArtifactMismatch("unknown artifact reference")
        try: raw = gzip.open(row[0], "rb").read()
        except OSError as e: raise ArtifactMismatch("artifact unreadable") from e
        if len(raw) != row[1] or hashlib.sha256(raw).hexdigest() != digest: raise ArtifactMismatch("artifact hash/length mismatch")
        return raw

    def append_record(self, dataset: Literal["A","B","C","preference"], record: RecordBase) -> None:
        if dataset not in RECORD_TYPES or not isinstance(record, RECORD_TYPES[dataset]): raise ValidationError("dataset/record type mismatch")
        payload = record.to_json()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cur = self.db.execute("INSERT OR IGNORE INTO events(dataset,payload) VALUES(?,?)", (dataset,payload))
            if cur.rowcount:
                line = json.dumps({"dataset":dataset,"payload":json.loads(payload)},sort_keys=True,separators=(",",":"))+"\n"
                with self.journal.open("a",encoding="utf-8") as fh: fh.write(line); fh.flush(); os.fsync(fh.fileno())
            self.db.execute("COMMIT")
        except Exception: self.db.execute("ROLLBACK"); raise

    def finalize_export(self) -> list[Path]:
        out = self.root / "datasets"; out.mkdir(exist_ok=True); paths=[]
        for dataset, name in DATASETS.items():
            rows = self.db.execute("SELECT payload FROM events WHERE dataset=? ORDER BY seq", (dataset,)).fetchall()
            target = out / f"{name}.jsonl"; tmp = target.with_suffix(".jsonl.tmp")
            with tmp.open("w",encoding="utf-8") as fh:
                for (payload,) in rows: fh.write(payload+"\n")
                fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp,target); paths.append(target)
        return paths

    def recover_journal(self) -> int:
        """Discard only torn journal tails; SQLite remains the commit authority."""
        if not self.journal.exists(): return 0
        good=[]
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            try:
                item=json.loads(line); RECORD_TYPES[item["dataset"]]; json.dumps(item["payload"]); good.append(line)
            except (ValueError, KeyError, json.JSONDecodeError, TypeError): break
        self.journal.write_text("\n".join(good)+("\n" if good else ""),encoding="utf-8")
        return len(good)
