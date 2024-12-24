# Clean everything
docker-compose down -v
docker system prune -a -f

# Rebuild and start
docker-compose up --build