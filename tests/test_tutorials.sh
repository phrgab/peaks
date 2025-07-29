#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TUTORIALS_DIR="$REPO_ROOT/tutorials"

max_retries=3
mkdir -p "$REPO_ROOT/tmp_outputs"

for nb in "$TUTORIALS_DIR"/*.ipynb; do
  echo "Running notebook: $nb"
  output="$REPO_ROOT/tmp_outputs/$(basename "$nb")"
  success=false

  for attempt in $(seq 1 $max_retries); do
    echo "Attempt $attempt for $nb..."
    set +e
    papermill "$nb" "$output" --log-output --progress-bar
    exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
      echo "Success on attempt $attempt"
      success=true
      break
    else
      echo "Notebook $nb failed on attempt $attempt with exit code $exit_code"
      sleep $((5 * attempt))
    fi
  done

  if [ "$success" = false ]; then
    echo "Notebook $nb failed after $max_retries attempts."
    exit 1
  fi
done
