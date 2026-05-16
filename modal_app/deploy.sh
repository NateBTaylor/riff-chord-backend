#!/usr/bin/env bash
# Deploy the Modal riff-pipeline app AND warm the snapshot in one step,
# so the first real user after the deploy doesn't pay the ~60s
# snapshot-creation cost.
#
# Usage (from the backend repo root):
#     bash modal_app/deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "▶ Deploying riff-pipeline to Modal..."
modal deploy modal_app/pipeline.py

echo ""
echo "▶ Warming snapshot (forces setup + creates the snapshot now,"
echo "  so the first user request hits a warm pool)..."
python3 modal_app/warm_snapshot.py

echo ""
echo "✓ Deploy + warm complete. New analyses will use the fresh snapshot."
