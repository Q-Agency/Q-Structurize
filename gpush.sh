#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "❌ Commit message required"
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi

COMMIT_MSG="$*"

echo "➕ git add ."
git add .

echo "💬 git commit -m \"$COMMIT_MSG\""
git commit -m "$COMMIT_MSG"

echo "📤 git push origin main"
git push origin main

echo "✅ Done."
