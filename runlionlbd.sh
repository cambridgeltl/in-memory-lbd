#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


#DATADIR="$SCRIPTDIR/data/complete"
#DATADIR="$SCRIPTDIR/data/neoplasms-1p"
#DATADIR="$SCRIPTDIR/data/neoplasms"
DATADIR="$SCRIPTDIR/data/neoplasms-old"

NODES="$DATADIR/nodes.csv"
#EDGES="$DATADIR/edges.csv"
EDGES="$DATADIR/edges-cut-0-4,16.csv"

python "$SCRIPTDIR"/runlionlbd.py \
       --nodes "$NODES" \
       --edges "$EDGES"
