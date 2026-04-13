#!/bin/bash
# Railway build script — runs after pip install
set -e

echo "==> Running migrations"
python manage.py migrate --noinput

echo "==> Collecting static files"
python manage.py collectstatic --noinput

echo "==> Creating output directory"
mkdir -p output

echo "==> Build complete"
