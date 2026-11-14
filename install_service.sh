#!/bin/bash
# Install and start person-counter-web as a systemd service.
# Usage: sudo ./install_service.sh
set -e

SERVICE=person-counter-web
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "Must run as root: sudo $0" >&2
    exit 1
fi

# Stop any manually started copy so it releases the camera and port 5001
pkill -u nvidia -f "python3 person_counter_web.py" || true
sleep 2

cp "$DIR/$SERVICE.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now "$SERVICE"

sleep 15
systemctl --no-pager status "$SERVICE" | head -12
curl -sI http://127.0.0.1:5001 | head -1
