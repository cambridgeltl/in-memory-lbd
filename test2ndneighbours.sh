#!/bin/bash

set -eu

if [ "$#" -lt 1 ]; then
    year=2017
else
    year="$1"
fi

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATADIR="$SCRIPTDIR/data/"
OUTDIR="$SCRIPTDIR/outputs/"

mkdir -p "$OUTDIR"

URL="http://127.0.0.1:8080/neighbours2"
#IDFILE="data/ids/1000-ids.txt"
NAME_ID_FILE="$DATADIR/Shamith-genes.txt"

metric="count"
types=Gene

cat "$NAME_ID_FILE" | while read l; do
    n=$(echo "$l" | awk '{ print $1 }')
    i=$(echo "$l" | awk '{ print $2 }')
    o="$OUTDIR/${n}-2nd-neighbours.PR.json"
    curl -sS "$URL/$i?year_metric=$metric&year=$year&type=$types" > "$o"
done

# cat "$IDFILE" | head -n 10 | while read i; do
#     curl -sS "$URL/$i?year_metric=$metric&year=$year&type=$types"
# done
