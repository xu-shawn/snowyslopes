#!/bin/bash
# scripts/run_origin.sh: the (0, 0) search, with MC-Measure run on every candidate it reports.
# usage: scripts/run_origin.sh NAME START THREADS [key=value ...]     (keys: see src/origin.cpp)
# Writes runs/NAME.csv (every seed that reaches the floods), runs/NAME.log (progress and candidates) and
# runs/NAME.verified (each candidate as MC-Measure measures it from block 0, 256, 0). Stop with Ctrl-C or by killing
# bin/origin: candidates still being measured are finished, and the log's last line gives the next index to start from.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 3 ] || { sed -n 3,6p "$0" | cut -c3-; exit 2; }
N=$1; S=$2; J=$3; shift 3
mkdir -p runs
bin/origin "$S" 0 "runs/$N" "$J" "$@" 2>&1 >/dev/null | (
    trap '' INT
    while read -r l; do
        echo "$l" >> "runs/$N.log"
        case "$l" in CANDIDATE*) set -- $l; echo "$2 $(bin/mcmeasure "$2" 0 0)" >> "runs/$N.verified" ;; esac
    done)
