#!/bin/bash

# Kill all running in-memory-lbd services.

SIGNAL="SIGTERM"

# Note: assumes that services are run from a directory matching the
# repository name and will miss any that are not.

ps aux \
    | egrep '[p]ython .*\bin-memory-lbd/runlionlbd.py' \
    | awk '{ print $2 }' \
    | while read pid; do
    echo "Sending $SIGNAL to $pid" >&2
    kill "-$SIGNAL" "$pid"
done

