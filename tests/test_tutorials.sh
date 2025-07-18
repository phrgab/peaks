#!/usr/bin/env bash
set -e

max_retries=3
mkdir -p tmp_outputs

for nb in tutorials/*.ipynb; do
  echo "Running notebook: $nb"
  output="tmp_outputs/$(basename "$nb")"
  success=false

  for attempt in $(seq 1 $max_retries); do
    echo "Attempt $attempt for $nb..."
    set +e  # allow papermill to fail without exiting script
    papermill "$nb" "$output" --log-output --progress-bar
    exit_code=$?
    set -e  # re-enable strict mode

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
