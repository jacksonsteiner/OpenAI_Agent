# Use the official Microsoft base image (includes the vscode user)
FROM mcr.microsoft.com/devcontainers/python:3.11

ARG OPENAI_API_KEY

# [Optional] Install system dependencies you might need
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
       build-essential curl git vim \
    && rm -rf /var/lib/apt/lists/*

# [Optional] Create virtualenv and install OpenAI SDK globally
RUN pip install --no-cache-dir openai

# [Optional] Set a default working directory
WORKDIR /workspaces/openai-agent-project

# Create vscode user if it doesn't exist
RUN id -u vscode &>/dev/null || useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode

# Switch to the non-root vscode user (already exists in this base image)
USER vscode
