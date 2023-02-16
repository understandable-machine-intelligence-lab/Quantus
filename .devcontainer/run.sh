#!/bin/zsh

set -e

PROJECT_DIR=$(realpath "$(pwd)"/..)

docker build -t devcontainer .
docker run --name devcontainer --mount type=bind,source="$PROJECT_DIR",target="/Quantus" --rm -it devcontainer