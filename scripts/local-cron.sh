#!/usr/bin/env bash
# Install / remove a launchd job that runs the pipeline on THIS Mac every few
# minutes. Facebook serves a login wall to datacenter IPs, so the scrape has to
# come from a residential connection - your machine is the simplest one.
#
#   bash scripts/local-cron.sh install [interval_seconds]   # default 600 (10 min)
#   bash scripts/local-cron.sh uninstall
#   bash scripts/local-cron.sh status
#
# Prereqs: the .venv exists (see README) and .env has NTFY_TOPIC.
set -euo pipefail

LABEL="com.autosniper.pipeline"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"

case "${1:-}" in
  install)
    interval="${2:-600}"
    [ -x "$PY" ] || { echo "no venv at $PY - create it first (README)"; exit 1; }
    [ -f "$REPO/.env" ] || echo "warning: $REPO/.env missing - notify won't send without NTFY_TOPIC"
    mkdir -p "$REPO/logs" "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO/scripts/run_once.sh</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHON</key><string>$PY</string>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StartInterval</key><integer>$interval</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$REPO/logs/pipeline.log</string>
  <key>StandardErrorPath</key><string>$REPO/logs/pipeline.err</string>
</dict>
</plist>
PLIST
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "installed: $LABEL every ${interval}s -> $REPO/logs/pipeline.log"
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed $LABEL"
    ;;
  status)
    launchctl list | grep "$LABEL" || echo "not loaded"
    ;;
  *)
    echo "usage: $0 {install [interval_seconds]|uninstall|status}"; exit 1 ;;
esac
