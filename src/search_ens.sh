#!/bin/bash
# Search for the ensemble subset that maximises mAP on the local val split.
set -u
cd /hy-tmp/cd || exit 1
OUT=runs/ens_search.log
: > "$OUT"

A="runs/q_s1024_s/weights/best.pt:1024"
B="runs/q_hsv/weights/best.pt:1024"
C="runs/q_mos05/weights/best.pt:1024"
D="runs/q_s1280_s/weights/best.pt:1280"
E="runs/q_combo/weights/best.pt:1024"
F="runs/q_combo_s1/weights/best.pt:1024"
G="runs/q_combo_m/weights/best.pt:1024"

try() {
  local tag="$1"; local members="$2"
  echo "### $tag" >> "$OUT"
  python3 tools/eval_ensemble.py --members "$members" --tta 1 --conf 0.01 2>/dev/null \
    | grep -E '^\{"members"' >> "$OUT"
}

try base4      "$B,$C,$D,$A"
try base4+m    "$B,$C,$D,$A,$G"
try base4+m+c  "$B,$C,$D,$A,$G,$E"
try base4+m+c2 "$B,$C,$D,$A,$G,$E,$F"
try trio       "$B,$C,$G"
try quad_m     "$B,$C,$D,$G"
try full7      "$A,$B,$C,$D,$E,$F,$G"
echo "### DONE" >> "$OUT"
