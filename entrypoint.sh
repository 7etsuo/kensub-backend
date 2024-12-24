#!/bin/bash

# Wait for PostgreSQL to be ready
while ! nc -z postgres 5432; do
  echo "Waiting for PostgreSQL to start..."
  sleep 1
done

echo "PostgreSQL started"

# Run Alembic migrations
# alembic upgrade head
# Execute the final command (either uvicorn or celery)
exec "$@"
