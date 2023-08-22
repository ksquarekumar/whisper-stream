# Base Image
# TODO: Change to something more suitable for NVIDIA/Inferencing
FROM python:3.10-slim AS base

# Main Layer
FROM base AS final
ARG STAGE

RUN apt-get update && \
    apt-get install -yq \
    --no-install-recommends \
    git \
    htop \
    curl \
    wget \
    build-essential

# Minimal copy of files needed to install Dependencies
COPY . /code

RUN python -m pip install --no-cache-dir --upgrade \
    pip wheel build sdist setuptools flit \
    && pip install --no-cache-dir --use-pep517 "/code/"

# Set common and PATH environment variables
ENV TZ=UTC \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    DEBIAN_FRONTEND=noninteractive

# Copy and install source package
WORKDIR /code

# hadolint ignore=DL3013
RUN pip install --no-cache-dir --no-deps "." \
    && apt-get remove build-essential -yq \
    && apt-get clean -yq \
    && apt-get autoclean  -yq \
    && apt-get autoremove --purge -yq

ENV STAGE=$STAGE
