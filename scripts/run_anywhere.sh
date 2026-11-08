#!/bin/bash
# scripts/run_anywhere.sh: the anywhere search, with MC-Measure run on every candidate it reports.
# usage: scripts/run_anywhere.sh NAME START THREADS [key=value ...]     (keys: see Params in src/anywhere.cpp)
# Writes runs/NAME.csv (every site with a patch), runs/NAME.worlds.csv (the best per gated seed), runs/NAME.log and
# runs/NAME.verified (each candidate as MC-Measure measures it at its block coordinates). Stop with Ctrl-C or by
# killing bin/anywhere: candidates still being measured are finished, and the log's last line gives the next seed.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 3 ] || { sed -n 3,6p "$0" | cut -c3-; exit 2; }
N=$1; S=$2; J=$3; shift 3
mkdir -p runs
bin/anywhere "$S" 0 "runs/$N" "$J" "$@" 2>&1 >/dev/null | (
    trap '' INT
    while read -r l; do
        echo "$l" >> "runs/$N.log"
        case "$l" in CANDIDATE*) set -- $l; echo "$2 $3 $4 $(bin/mcmeasure "$2" "$3" "$4")" >> "runs/$N.verified" ;; esac
    done)
