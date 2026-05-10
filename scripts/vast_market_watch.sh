#!/usr/bin/env bash
# Periodic vast.ai marketplace scan.
# Emits a stdout line ONLY when a deal appears that is BOTH:
#   - bid < $0.10/hr (absolute budget cap)
#   - >=25% cheaper per effective-core than what we're currently paying
#     (raised from 10% on 2026-05-10: vast.ai search `min_bid` is the host
#      floor, not the live auction price. Sub-25% deals can't usually be won.)
# No minimum core floor — small cheap boxes are fair game.
set -uo pipefail

CUR_BID="${CUR_BID:-0.10}"           # current hourly bid
CUR_EFF="${CUR_EFF:-24}"             # current effective cores
INTERVAL="${INTERVAL:-600}"          # seconds between scans
MAX_BID="${MAX_BID:-0.10}"           # absolute hourly budget cap
KEY="${VAST_AI_API_KEY:-$(grep '^export VAST_AI_API_KEY=' ~/.zshrc | sed 's/^export VAST_AI_API_KEY=//' | tr -d '"')}"

while true; do
  python3 - <<PY 2>/dev/null
import json, os, subprocess, sys
env = os.environ.copy(); env["VAST_API_KEY"] = "$KEY"
q = "cpu_cores_effective>=8 cpu_ram>=8 reliability>0.95 inet_down>=200 dph<0.20 verified=true rentable=true gpu_ram>=4"
try:
    out = subprocess.check_output(["vastai","search","offers",q,"--order","min_bid","--raw"], env=env, timeout=25)
    d = json.loads(out)
except Exception as e:
    print(f"[market-watch] scan failed: {e}")
    sys.exit(0)

# Filter Blackwell out (cc>=1200) — needs torch>=2.6
nonbw = [o for o in d if o.get('compute_cap',0) < 1200]
cur_p_per_core = $CUR_BID / max(1, $CUR_EFF)

best = None
for o in nonbw:
    eff = o.get('cpu_cores_effective', 0)
    bid = o.get('min_bid', 0)
    if eff <= 0 or bid <= 0: continue
    # Hard cap: absolute bid below \$$MAX_BID
    if bid > $MAX_BID: continue
    p = bid / eff
    # Must be >=25% cheaper per core
    if p > cur_p_per_core * 0.75: continue
    if best is None or p < best['p']:
        best = {'p': p, 'o': o}

if best:
    o = best['o']
    print(f"[better deal] id={o['id']}  bid=\${o['min_bid']:.3f}  eff={o['cpu_cores_effective']:.0f}c  gpu={o.get('gpu_name','?')[:18]}  loc={(o.get('geolocation') or '?')[:8]} — \${best['p']*1000:.2f}/1k-core-hr (vs current \${cur_p_per_core*1000:.2f}, {(1-best['p']/cur_p_per_core)*100:.0f}% cheaper)")
PY
  sleep $INTERVAL
done
