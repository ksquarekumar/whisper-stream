<!--
# Copyright © 2023 krishnakumar <ksquarekumar@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at:
#
# https://github.com/ksquarekumar/whisper-stream/blob/main/LICENSE
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
#
# This file is part of the whisper-stream.
# see (https://github.com/ksquarekumar/whisper-stream)
#
# SPDX-License-Identifier: Apache-2.0
#
# You should have received a copy of the APACHE LICENSE, VERSION 2.0
# along with this program. If not, see <https://apache.org/licenses/LICENSE-2.0>
-->

# Serving `Faster-Whisper`

## Preparation

> add an appropriate `.env` & `log_config.yaml` file to the working directory, use of both are optional, but highly recommended.
>
> > refer to [.test.env](./.test.env) and [log_config.yaml](./log_config.yaml) samples in [source](.) directory
>
> > for more values that can be put in `.env` file, refer to [FasterWhisperAPILaunchConfig](../../whisper_stream/projects/faster_whisper_api/config/faster_whisper_api_launch_config.py)

## Install

- from `repo root`

```shell
poetry install --all-extras
```

- from `project root`

```shell
poetry install
```

## Launch

```shell
$ > launch_faster_whisper_api
```
