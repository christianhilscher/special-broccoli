#!/bin/bash

# Environment variables
SSH_HOST=$SSH_HOST
CONTAINER_REPO=$CONTAINER_REPO
VERSION=$VERSION

# Stop all containers
ssh -o StrictHostKeyChecking=no $SSH_HOST "docker stop $(docker ps -aq)"

# Pull and start specific version
ssh -o StrictHostKeyChecking=no $SSH_HOST "docker pull $CONTAINER_REPO:$VERSION"
ssh -o StrictHostKeyChecking=no $SSH_HOST "docker run -d $CONTAINER_REPO:$VERSION"
