#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATADIR="$SCRIPTDIR/data/complete"
#DATADIR="$SCRIPTDIR/data/neoplasms"
#DATADIR="$SCRIPTDIR/data/neoplasms-1p"
NODES="$DATADIR/nodes.csv"
EDGES="$DATADIR/edges.csv"

python "$SCRIPTDIR"/runlionlbd.py \
       --nodes "$NODES" \
       --edges "$EDGES"
