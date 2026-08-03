#!/usr/bin/env bash
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run with sudo" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the host available for the trading agent while connected to AC power.
systemsetup -setsleep Never
systemsetup -setwakeonnetworkaccess on
systemsetup -setrestartpowerfailure on
systemsetup -setrestartfreeze on

"$SCRIPT_DIR/install-macos.sh"

echo
echo "Power settings:"
pmset -g custom
echo
echo "FileVault status:"
fdesetup status
