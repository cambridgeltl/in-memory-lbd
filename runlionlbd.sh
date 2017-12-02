#!/bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATADIR="/srv/lion/neo4j-import/complete"

NODES="$DATADIR/nodes.csv"
EDGES="$DATADIR/edges.csv"
META="$DATADIR/meta.csv"

python "$SCRIPTDIR"/runlionlbd.py \
       --nodes "$NODES" \
       --edges "$EDGES" \
       --meta "$META"
