#!/bin/bash

# Special-purpose script for comparing the outputs of different LBD
# implementations for Shamith's experiment.

set -eu

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR1="$SCRIPTDIR/outputs"
DIR2="/srv/lion/results/for-Shamith/051017/"

function print_score_and_id {
    python -c '
import sys
import json
seen = set()
for d in json.load(open("'"$1"'")):
    if d["'$2'"] not in seen:
        print d["comp"], d["'$2'"]
        seen.add(d["'$2'"])
'
}

for f1 in "$DIR1/"*-1st-neighbours.PR.json; do
    f2="$DIR2/"$(basename "$f1")
    echo -n "diff "$(basename "$f1")": "
    diff <(print_score_and_id "$f1" "B" | sort -rn) \
	 <(print_score_and_id "$f2" "B" | sort -rn) | wc -l
done

for f1 in "$DIR1/"*-2nd-neighbours.PR.json; do
    f2="$DIR2/"$(basename "$f1")
    echo -n "diff "$(basename "$f1")": "
    diff <(print_score_and_id "$f1" "C" | sort -rn) \
	 <(print_score_and_id "$f2" "C" | sort -rn) | wc -l
done
