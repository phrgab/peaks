#!/usr/bin/env bash
set -e  # Exit on any error

# --- Resolve script dir (so all paths are relative to it) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$SCRIPT_DIR"
TUTORIALS_SRC="$SCRIPT_DIR/../tutorials"
TUTORIALS_DST="$SCRIPT_DIR/source/_tutorials"

# --- Parse options ---
CLEAN=false
FORCE_EXEC=false

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=true ;;
        --exec-nb) FORCE_EXEC=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# --- Copy tutorials ---
echo "Copying tutorials from $TUTORIALS_SRC to $TUTORIALS_DST"
mkdir -p "$TUTORIALS_DST"
cp -fr "$TUTORIALS_SRC"/* "$TUTORIALS_DST/"

# --- Clean build if requested ---
if [ "$CLEAN" = true ]; then
    echo "Running: make clean"
    make -C "$DOCS_DIR" clean
fi

# --- Set env for forcing execution ---
if [ "$FORCE_EXEC" = true ]; then
    echo "Notebook execution forced (nb_execution_mode=force)"
    export FORCE_NB_EXECUTION=1
else
    unset FORCE_NB_EXECUTION
fi

# --- Build docs ---
echo "Building HTML docs..."
make -C "$DOCS_DIR" html

# --- Cleanup copied tutorials ---
echo "Cleaning up copied tutorials..."
rm -rf "$TUTORIALS_DST"

echo "Docs build complete."