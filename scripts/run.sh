#!/bin/bash
# scripts/run.sh
# Allow X11 forwarding
xhost +local:docker

# Run containers
docker-compose up

# Revoke X11 after exit
xhost -local:docker
