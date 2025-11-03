#!/bin/bash
# view_latest_logs.sh
# View newest .out/.err files, or delete all but the most recent two with --delete

set -euo pipefail
shopt -s nullglob

usage() {
  cat <<'EOF'
Usage:
  view_latest_logs.sh            # interactively view one of the 4 newest logs
  view_latest_logs.sh --delete   # delete all *.out/*.err except the most recent 2
  view_latest_logs.sh -h|--help  # show this help
EOF
}

# Parse optional flag
DELETE_MODE=false
case "${1-}" in
  --delete) DELETE_MODE=true ;;
  -h|--help) usage; exit 0 ;;
  "" ) : ;; # no args
  * ) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
esac

# Collect *.out and *.err files, newest first
# Using ls -t for time sort; readarray preserves filenames with spaces
readarray -t all_files < <(ls -t *.out *.err 2>/dev/null || true)

if [ ${#all_files[@]} -eq 0 ]; then
  echo "No .out or .err files found in the current directory."
  exit 1
fi

if $DELETE_MODE; then
  # Keep only the 2 newest across both extensions
  keep=("${all_files[@]:0:2}")
  to_delete=("${all_files[@]:2}")

  if [ ${#to_delete[@]} -eq 0 ]; then
    echo "Nothing to delete. Fewer than 3 matching files exist."
    exit 0
  fi

  echo "Keeping the 2 most recent:"
  printf '  %s\n' "${keep[@]}"
  echo
  echo "Deleting ${#to_delete[@]} file(s):"
  printf '  %s\n' "${to_delete[@]}"
  echo
  read -r -p "Proceed? [y/N] " ans
  case "$ans" in
    [Yy]* )
      rm -- "${to_delete[@]}"
      echo "Deletion complete."
      ;;
    * )
      echo "Aborted. No files deleted."
      ;;
  esac
  exit 0
fi

# ---- Interactive view mode (default) ----

# Show only the 4 newest (mixed .out/.err)
files=("${all_files[@]:0:4}")

# Color codes
BOLD="\033[1m"
GREEN="\033[32m"
RESET="\033[0m"

echo "Select a file to open:"
i=1
for file in "${files[@]}"; do
  if [ -e "$file" ]; then
    # Get modification date/time without seconds (GNU/BSD stat compatibility)
    if stat --version &>/dev/null; then
      mod_time=$(stat -c '%y' "$file" | cut -d'.' -f1 | awk '{print $1, substr($2,1,5)}')
    else
      mod_time=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$file")
    fi

    if [ $i -le 2 ]; then
      echo -e "  [$i] ${BOLD}${GREEN}$file  —  $mod_time${RESET}"
    else
      echo "  [$i] $file  —  $mod_time"
    fi
    ((i++))
  fi
done

read -p "Enter number: " choice
if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#files[@]} ]; then
  file="${files[$((choice-1))]}"
  echo "Opening $file..."
  if command -v bat &>/dev/null; then
    bat "$file"
  else
    less "$file"
  fi
else
  echo "Invalid selection."
fi