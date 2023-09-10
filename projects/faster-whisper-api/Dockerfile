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
# TODO: Change to something more suitable for NVIDIA/Inferencing
FROM python:3.11-slim AS base

# Main Layer
FROM base AS final
ARG STAGE


# hadolint ignore=DL3008,DL3009
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


# hadolint ignore=DL3013
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
    && apt-get autoremove --purge -yq \
    && rm -rf /var/lib/apt/lists/*

ENV STAGE=$STAGE
