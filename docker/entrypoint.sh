#!/bin/bash

echo "Loading configuration from Vault"
cat /etc/secrets/env.txt > .env
cat /etc/secrets/env.txt > /app/.env
cd /app
#uvicorn server.api.scanner.views:app --host 0.0.0.0 --port 8014
python3 run.py
exec "$@"
