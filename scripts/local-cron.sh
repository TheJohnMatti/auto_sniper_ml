#!/usr/bin/env bash
# Install / remove the launchd jobs that run the pipeline on THIS Mac. Facebook
# serves a login wall to datacenter IPs, so the scrape has to come from a
# residential connection - your machine is the simplest one.
#
#   bash scripts/local-cron.sh install [scan_secs] [enrich_secs]   # default 300 / 1200
#   bash scripts/local-cron.sh uninstall
#   bash scripts/local-cron.sh status
#
# Two jobs (they share a single-flight lock, so they never overlap):
#   com.autosniper.scan    every ~5 min   scrape feed -> score -> notify (fast)
#   com.autosniper.enrich  every ~20 min  fetch descriptions -> re-score -> notify
#
# Prereqs: the .venv exists (see README) and .env has NTFY_TOPIC.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
AGENTS="$HOME/Library/LaunchAgents"
LEGACY="com.autosniper.pipeline"   # the old single job, removed on install

write_plist() {   # label  interval  env_key=env_val ...
  local label="$1" interval="$2"; shift 2
  local envxml=""
  for kv in "$@"; do
    envxml+="    <key>${kv%%=*}</key><string>${kv#*=}</string>"$'\n'
  done
  cat > "$AGENTS/$label.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/scripts/run_once.sh</string></array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHON</key><string>$PY</string>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
$envxml  </dict>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StartInterval</key><integer>$interval</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$REPO/logs/$label.log</string>
  <key>StandardErrorPath</key><string>$REPO/logs/$label.err</string>
</dict>
</plist>
PLIST
  launchctl unload "$AGENTS/$label.plist" 2>/dev/null || true
  launchctl load "$AGENTS/$label.plist"
  echo "installed $label every ${interval}s -> logs/$label.log"
}

case "${1:-}" in
  install)
    scan="${2:-300}"; enrich="${3:-1200}"
    [ -x "$PY" ] || { echo "no venv at $PY - create it first (README)"; exit 1; }
    [ -f "$REPO/.env" ] || echo "warning: $REPO/.env missing - notify won't send without NTFY_TOPIC"
    mkdir -p "$REPO/logs" "$AGENTS"
    launchctl unload "$AGENTS/$LEGACY.plist" 2>/dev/null || true
    rm -f "$AGENTS/$LEGACY.plist"
    write_plist com.autosniper.scan   "$scan"   "SKIP_DESCRIPTIONS=1"
    write_plist com.autosniper.enrich "$enrich" "SKIP_SCRAPE=1"
    ;;
  uninstall)
    for l in com.autosniper.scan com.autosniper.enrich "$LEGACY"; do
      launchctl unload "$AGENTS/$l.plist" 2>/dev/null || true
      rm -f "$AGENTS/$l.plist"
    done
    echo "removed autosniper launchd jobs"
    ;;
  status)
    launchctl list | grep autosniper || echo "not loaded"
    ;;
  *)
    echo "usage: $0 {install [scan_secs] [enrich_secs]|uninstall|status}"; exit 1 ;;
esac
