##
## # Copyright © 2023 krishnakumar <ksquarekumar@gmail.com>.
## #
## # Licensed under the Apache License, Version 2.0 (the "License"). You
## # may not use this file except in compliance with the License. A copy of
## # the License is located at:
## #
## # https://github.com/ksquarekumar/whisper-stream/blob/main/LICENSE
## #
## # or in the "license" file accompanying this file. This file is
## # distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
## # ANY KIND, either express or implied. See the License for the specific
## # language governing permissions and limitations under the License.
## #
## # This file is part of the whisper-stream.
## # see (https://github.com/ksquarekumar/whisper-stream)
## #
## # SPDX-License-Identifier: Apache-2.0
## #
## # You should have received a copy of the APACHE LICENSE, VERSION 2.0
## # along with this program. If not, see <https://apache.org/licenses/LICENSE-2.0>
##

# Base Image
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Builder Stage
FROM base AS builder
ARG APP_ENV="development"

# Set common and PATH environment variables
ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008,DL3009,DL3013
RUN apt-get update \
    && apt-get install -yq \
    --no-install-recommends \
    software-properties-common \
    python3-pip \
    python3-distutils \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -yq \
    --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2 \
    && python -m pip install --no-cache-dir --upgrade \
    pip wheel python-build sdist setuptools flit

# Minimal copy of files needed to install Dependencies for all stages
COPY . /code

# Set default work-directory for all stages
WORKDIR /code/projects/faster-whisper-api

# Temporary wheel builder stage
FROM builder as wheelbuilder
ARG PACKAGE_NAMESPACE="whisper_stream"
ARG PACKAGE_VERSION="0.0.5"

# Install build dependencies and build wheel
# hadolint ignore=DL3013,DL3042
RUN python -m pip install poetry \
    && poetry self add poetry-multiproject-plugin \
    && mkdir /code/projects/faster-whisper-api/dist \
    && poetry build-project --format=wheel --with-top-namespace "${PACKAGE_NAMESPACE}"

# Final app stage
From builder as final
ARG PACKAGE_NAMESPACE="whisper_stream"
ARG PACKAGE_NAME="faster_whisper_api"
ARG PACKAGE_VERSION="0.0.5"
COPY --from=wheelbuilder /code/projects/faster-whisper-api/dist/*.whl /code/projects/faster-whisper-api/

# hadolint ignore=DL3013
RUN python -m pip install --no-cache-dir --use-pep517 -r "requirements.txt"

# hadolint ignore=DL3013
RUN python -m pip install --no-cache-dir --no-deps "${PACKAGE_NAMESPACE}_${PACKAGE_NAME}-${PACKAGE_VERSION}-py3-none-any.whl" \
    && python -m pip install --no-cache-dir --no-deps "${PACKAGE_NAMESPACE}_${PACKAGE_NAME}" \
    && apt-get clean -yq \
    && apt-get autoclean  -yq \
    && apt-get autoremove --purge -yq \
    && rm -rf /var/lib/apt/lists/*

ENV APP_ENV=$APP_ENV
