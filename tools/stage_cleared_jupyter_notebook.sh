#!/bin/bash
# Stage the cleared changes in a Jupyter Notebook:
#   - Save the current state
#   - Clear the notebook
#   - Stage the cleared state
#   - Restore original changes unstaged

paths=("${@}")

git add "${paths[@]}" || exit 1
git stash -k || exit 1
./tools/clear_jupyter_notebook.sh "${paths[@]}" || {
    echo "error while clearing notebooks" >&2
    exit 1
}
git add "${paths[@]}" || exit 1
git restore --source=stash@{0} --worktree -- . || exit 1
git stash drop || exit 1
