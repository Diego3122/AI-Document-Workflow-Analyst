#!/usr/bin/env bash
set -euo pipefail

umask 0002
mkdir -p /home/site/data/uploads /home/site/data/chroma /home/site/data/logs /home/site/data/evals
exec /usr/bin/supervisord -c /app/supervisord.conf
