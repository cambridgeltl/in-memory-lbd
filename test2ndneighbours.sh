#!/bin/bash

URL="http://127.0.0.1:8080/neighbours2"
IDFILE="data/ids/1000-ids.txt"

metric="count"
year=2002
types=Gene

cat "$IDFILE" | head -n 10 | while read i; do
    curl -sS "$URL/$i?year_metric=$metric&year=$year&type=$types"
done
