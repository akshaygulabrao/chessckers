"""Run identity for cc command headers: the human-meaningful RUN_NAME."""
import os
import sqlite3

_DB = os.path.join(os.path.dirname(__file__), "..", "..", "lczero-server", "chessckers.db")

def run_name(db: str | None = None) -> str:
    """Current run's name ('' if no DB yet). DB id is always 1; the name is the identity."""
    try:
        con = sqlite3.connect(f"file:{db or _DB}?mode=ro", uri=True)
        row = con.execute("SELECT description FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
        con.close()
        return row[0] if row and row[0] else ""
    except Exception:
        return ""
