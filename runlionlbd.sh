#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATADIR="$SCRIPTDIR/data/neoplasms"
NODES="$DATADIR/nodes.csv"
EDGES="$DATADIR/edges.csv"

python "$SCRIPTDIR"/runlionlbd.py \
       --nodes "$NODES" \
       --edges "$EDGES"
