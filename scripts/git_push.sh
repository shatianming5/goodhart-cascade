#!/bin/bash
# Push current state to GitHub
# Usage: ./scripts/git_push.sh "commit message"

set -e

cd "$(dirname "$0")/.."

MSG=${1:-"Update experiment results"}

# Configure git if needed
git config user.email "shatianming5@users.noreply.github.com" 2>/dev/null || true
git config user.name "shatianming5" 2>/dev/null || true

# Stage all tracked changes (but skip large model files)
git add -A
git reset -- '*.bin' '*.safetensors' '*.pt' '*.pth' '*.onnx' 2>/dev/null || true

# Check if there are changes
if git diff --cached --quiet; then
    echo "No changes to commit"
    exit 0
fi

git commit -m "$MSG"

# Push using token
git push https://shatianming5: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@github.com/shatianming5/goodhart-cascade.git HEAD:main

echo "Pushed to GitHub: $MSG"
