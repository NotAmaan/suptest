#!/bin/bash

# Local build and push script for SUPIR-Demo

# Configuration
REGISTRY="ghcr.io"
GITHUB_USERNAME="notamaan"  # TODO: Replace with your GitHub username
REPO_NAME="suptest"
IMAGE_NAME="$REGISTRY/$GITHUB_USERNAME/$REPO_NAME"

# Get version from git tag or use 'latest'
VERSION=$(git describe --tags --always 2>/dev/null || echo "latest")

echo "Building Docker image: $IMAGE_NAME:$VERSION"

# Build the image
docker build -t "$IMAGE_NAME:$VERSION" -t "$IMAGE_NAME:latest" .

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"

# Ask to push
read -p "Push to GitHub Container Registry? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Logging in to GitHub Container Registry..."
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USERNAME" --password-stdin
    
    if [ $? -ne 0 ]; then
        echo "Login failed! Make sure GITHUB_TOKEN is set"
        exit 1
    fi
    
    echo "Pushing $IMAGE_NAME:$VERSION..."
    docker push "$IMAGE_NAME:$VERSION"
    docker push "$IMAGE_NAME:latest"
    
    echo "Push complete!"
    echo "Image available at: $IMAGE_NAME:$VERSION"
else
    echo "Skipping push"
fi