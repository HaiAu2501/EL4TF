# Reference: https://docs.docker.com/reference/dockerfile/
# Using uv in Docker: https://docs.astral.sh/uv/guides/integration/docker
FROM ubuntu:24.04 AS devcontainer

RUN apt-get update && apt-get install -y \
    git \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv

RUN python3 -m venv /venv

COPY requirements.txt /tmp/
RUN /venv/bin/pip install -r /tmp/requirements.txt
