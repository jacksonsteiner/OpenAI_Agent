# Use the official Microsoft base image (includes the vscode user)
FROM mcr.microsoft.com/devcontainers/python:3.11

ARG OPENAI_API_KEY

# [Optional] Install system dependencies you might need
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       git \
       vim \
       ca-certificates \
       gnupg \
       lsb-release \
       wget \
    && rm -rf /var/lib/apt/lists/*

# Add HashiCorp GPG key
RUN wget -O - https://apt.releases.hashicorp.com/gpg \
    | gpg --dearmor \
    | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg > /dev/null

# Add HashiCorp repo
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
    https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/hashicorp.list

# Install Terraform
RUN apt-get update && apt-get install -y terraform \
    && rm -rf /var/lib/apt/lists/*

# Add Azure-CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# [Optional] Create virtualenv and install OpenAI SDK globally
RUN pip install --no-cache-dir openai

# [Optional] Set a default working directory
WORKDIR /workspaces/openai-agent-project

# Create vscode user if it doesn't exist
RUN id -u vscode &>/dev/null || useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode

# Switch to the non-root vscode user (already exists in this base image)
USER vscode
