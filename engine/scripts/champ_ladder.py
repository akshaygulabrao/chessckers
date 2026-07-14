#!/usr/bin/env python3
"""cc champs — ladder the gate's ACTUAL champions + candidates (server .bin nets).

`cc ladder` plays the trainer's iter-async-*.pt checkpoints — a strength curve of
the raw training net, but NOT the nets the gate saw (those are EMA publishes,
gated 40-games-vs-best, stored server-side). This audits the gate itself: read
the promotion history from the server DB, pull the crowned champion + log-spaced
past champions + the most recent REJECTED candidates from networks/<sha> (stored
gzipped), gunzip them to .bin, and hand them to ladder.py in --engine mode (these
nets exist only as fork-loadable .bin — there is no .pt, so no MCTS mode).

Field (deduped, labels show the fleet-visible net number):
  best   the server's crowned champion (training_runs.best_network_id)
  cN     past champions, log-spaced back from best (1,2,4,8,... promotions ago —
         mirrors the league pool sampling in main.go)
  rN     newest rejected candidates (did the gate wrongly turn one away?)

  cc champs                        # 5 past champs + 3 rejected + best, 12 games/pair
  cc champs --champs 3 --cands 2 --games 20
  cc champs --list                 # print promotion history + field, play nothing
  cc champs --jsonl champs_audit.jsonl   # nightly cron: also append one audit row
options: --db PATH  --run-id 1  --champs N  --cands N  --list  --since-net N  --all-runs
         --jsonl PATH  + every ladder option passes through (--games --visits ...)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import _run_ident  # noqa: E402  (run name for the audit row)
import ladder  # noqa: E402  (reuses _SERVER_DIR + the whole match loop / rendering)
from engine_uci import DEFAULT_BINARY  # noqa: E402

_SERVER_DIR = ladder._SERVER_DIR


def _history(db: str, run_id: int):
    """Promotion history for one training run: (num map, best_id, match rows).
    Same filters as strength.py: done, real promotion matches, not soft-deleted."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT best_network_id FROM training_runs WHERE id=?",
                      (run_id,)).fetchone()
    if not row or not row[0]:
        raise SystemExit(f"champs: training run {run_id} has no best network ({db})")
    num = {r["id"]: r["network_number"]
           for r in con.execute("SELECT id, network_number FROM networks")}
    sha = {r["id"]: (r["sha"], r["path"])
           for r in con.execute("SELECT id, sha, path FROM networks")}
    rows = con.execute(
        "SELECT id, candidate_id, current_best_id, wins, losses, draws, passed "
        "FROM matches WHERE training_run_id=? AND done=1 AND test_only=0 "
        "AND special_params=0 AND deleted_at IS NULL ORDER BY id",
        (run_id,)).fetchall()
    con.close()
    return num, sha, row[0], rows


def _scope_to_current_run(rows, num, since_net):
    """A buffer-preserving scale-up reuses training_run 1, so the DB can hold a
    PRIOR run's nets/matches (wrong arch — would SIGTRAP the fork). Scope to the
    current run: it starts at the most recent BOOTSTRAP-promoted net (was best
    but never a match candidate). Same detection as strength.py."""
    if since_net is None:
        cand_ids = {r["candidate_id"] for r in rows}
        boots = [i for i in ({r["current_best_id"] for r in rows} - cand_ids) if i in num]
        if not boots:
            return rows
        since_net = num[max(boots, key=lambda i: num[i])]
    return [r for r in rows if num.get(r["candidate_id"], -1) >= since_net]


def _pick_field(rows, best_id, n_champs, n_cands):
    """[(label, network_id)] oldest-context-first, best last. Champions =
    bootstrap best + each passed candidate; sample log-spaced back from best
    (1,2,4,8,... promotions ago, the league-pool spacing). Rejected = newest
    done-but-failed candidates."""
    champions = []
    if rows:
        champions.append(rows[0]["current_best_id"])  # the bootstrap champion
    champions += [r["candidate_id"] for r in rows if r["passed"]]
    if not champions or champions[-1] != best_id:
        champions.append(best_id)  # bootstrap-only run, or history/DB drift

    picked: list[int] = []
    d = 1
    while len(picked) < n_champs and d < len(champions):
        c = champions[-1 - d]
        if c != best_id and c not in picked:
            picked.append(c)
        d *= 2
    rejected = []
    for r in reversed(rows):
        if not r["passed"] and r["candidate_id"] not in rejected \
                and r["candidate_id"] != best_id and r["candidate_id"] not in picked:
            rejected.append(r["candidate_id"])
        if len(rejected) >= n_cands:
            break
    return picked, rejected


def _materialize(net_id: int, label: str, sha_path, out_dir: str) -> str:
    """Gunzip networks/<sha> to <out_dir>/<label>.bin (ladder labels = filename).
    Server stores the bridge's gzipped upload verbatim; sniff the magic bytes."""
    sha, rel = sha_path[net_id]
    src = os.path.join(_SERVER_DIR, rel or os.path.join("networks", sha))
    raw = open(src, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if len(raw) < 1024:
        raise SystemExit(f"champs: {src} is implausibly small ({len(raw)}B) — corrupt?")
    out = os.path.join(out_dir, f"{label}.bin")
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "wb") as f:
        f.write(raw)
    return out


def _append_audit(jsonl: str, json_out: str, run: str, best_net: int | None) -> None:
    """Append one audit row from ladder's --json-out result — the nightly-cron trail
    answering 'is best actually on top, and is the field spreading?'. Non-fatal: an
    audit failure must never fail the ladder run that just played."""
    try:
        with open(json_out) as f:
            res = json.load(f)
        labels, elos, ng = res["labels"], res["elo"], res["n_games"]
        order = sorted(range(len(labels)), key=lambda i: -elos[i])
        rank = next((k for k, i in enumerate(order, 1) if labels[i] == "best"), None)
        row = {"ts": time.time(), "run": run, "best_net": best_net, "field": labels,
               "elo": [round(e, 1) for e in elos],
               "spread": round(max(elos) - min(elos), 1), "best_rank": rank,
               "games_per_pair": max((g for r in ng for g in r), default=0)}
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"champs-audit: spread={row['spread']:.0f} best_rank={rank}/{len(labels)}"
              f"  → {jsonl}")
    except Exception as e:  # noqa: BLE001
        print(f"champs-audit: FAILED (non-fatal): {e}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ladder the gate's real champions/candidates (server .bin nets).")
    ap.add_argument("--db", default=os.path.join(_SERVER_DIR, "chessckers.db"))
    ap.add_argument("--run-id", type=int, default=1, help="training run id (always 1)")
    ap.add_argument("--champs", type=int, default=5,
                    help="log-spaced past champions to include (default 5)")
    ap.add_argument("--cands", type=int, default=3,
                    help="newest REJECTED candidates to include (default 3)")
    ap.add_argument("--since-net", type=int, default=None,
                    help="scope to candidates with network_number >= N (default: auto-"
                         "detect the current run's bootstrap boundary)")
    ap.add_argument("--all-runs", action="store_true",
                    help="disable current-run scoping (DANGER: an old run's nets may "
                         "have a different arch and SIGTRAP the fork)")
    ap.add_argument("--out-dir", default=os.path.join(_SERVER_DIR, "networks", "_ladder"),
                    help="where the gunzipped .bin nets go")
    ap.add_argument("--engine", default=DEFAULT_BINARY,
                    help="fork binary (always engine mode — server nets are .bin-only)")
    ap.add_argument("--list", action="store_true",
                    help="print promotion history + field and exit (no games)")
    ap.add_argument("--jsonl", default="",
                    help="append one audit row {ts, run, best_net, field, elo, spread, "
                         "best_rank, games_per_pair} here after the ladder (cron mode)")
    args, ladder_args = ap.parse_known_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"champs: db not found: {args.db} (pass --db)")
    num, sha_path, best_id, rows = _history(args.db, args.run_id)
    if not args.all_runs:
        rows = _scope_to_current_run(rows, num, args.since_net)
    picked, rejected = _pick_field(rows, best_id, args.champs, args.cands)

    n_prom = sum(1 for r in rows if r["passed"])
    print(f"champs: {len(rows)} gate matches, {n_prom} promotions | "
          f"best = net #{num.get(best_id, '?')} (id {best_id})")
    print("  recent history (cand W-L-D vs best -> verdict):")
    for r in rows[-10:]:
        print(f"    #{num.get(r['candidate_id'], '?'):>4} "
              f"{r['wins']}-{r['losses']}-{r['draws']} vs #{num.get(r['current_best_id'], '?'):<4}"
              f" -> {'PROMOTED' if r['passed'] else 'rejected'}")

    field = ([(f"c{num[i]}", i) for i in sorted(picked, key=lambda i: num[i])]
             + [(f"r{num[i]}", i) for i in sorted(rejected, key=lambda i: num[i])]
             + [("best", best_id)])
    print(f"  field: {', '.join(lbl for lbl, _ in field)}")
    if args.list:
        return 0

    bins = [_materialize(i, lbl, sha_path, args.out_dir) for lbl, i in field]
    json_out = ""
    if args.jsonl:  # route ladder's result matrix through a temp --json-out for the audit
        fd, json_out = tempfile.mkstemp(suffix=".json", prefix="champs-")
        os.close(fd)
        ladder_args = [*ladder_args, "--json-out", json_out]
    # Hand off to ladder's match loop / Elo / rendering: server nets are raw .bin,
    # so force engine mode with an explicit binary (never bare --engine).
    sys.argv = ["ladder.py", *bins, "--engine", args.engine, *ladder_args]
    rc = ladder.main()
    if args.jsonl:
        _append_audit(args.jsonl, json_out, _run_ident.run_name(args.db), num.get(best_id))
        try:
            os.unlink(json_out)
        except OSError:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
