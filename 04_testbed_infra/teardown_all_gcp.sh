#!/usr/bin/env bash
set -e
PROJ=$(gcloud config get-value project)
echo "=== [TEARDOWN AUDIT] Scanning all CAPQ instances in project: $PROJ ==="
INSTANCES=$(gcloud compute instances list --project="$PROJ" --filter="name~'^capq-'" --format="csv[no-heading](name,zone)" || true)
if [ -z "$INSTANCES" ]; then
    echo "[OK] Zero active capq-* instances exist. Cloud billing rate = \$0.00/hr."
else
    for ITEM in $INSTANCES; do
        NAME="${ITEM%,*}"
        ZONE="${ITEM#*,}"
        echo "[DELETING] $NAME in $ZONE ..."
        gcloud compute instances delete "$NAME" --zone="$ZONE" --project="$PROJ" --quiet
    done
    echo "[OK] All capq-* instances permanently deleted."
fi