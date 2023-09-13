#!python
#
# # Copyright © 2023 krishnakumar <ksquarekumar@gmail.com>.
# #
# # Licensed under the Apache License, Version 2.0 (the "License"). You
# # may not use this file except in compliance with the License. A copy of
# # the License is located at:
# #
# # https://github.com/ksquarekumar/whisper-stream/blob/main/LICENSE
# #
# # or in the "license" file accompanying this file. This file is
# # distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# # ANY KIND, either express or implied. See the License for the specific
# # language governing permissions and limitations under the License.
# #
# # This file is part of the whisper-stream.
# # see (https://github.com/ksquarekumar/whisper-stream)
# #
# # SPDX-License-Identifier: Apache-2.0
# #
# # You should have received a copy of the APACHE LICENSE, VERSION 2.0
# # along with this program. If not, see <https://apache.org/licenses/LICENSE-2.0>
#
from fastapi import FastAPI
from whisper_stream.projects.faster_whisper_api.di.common import (
    CommonServiceContainer,
)
from whisper_stream.projects.faster_whisper_api.di.faster_whisper_model import (
    FasterWhisperModelServiceContainer,
)
from whisper_stream.projects.faster_whisper_api.app import (
    create_app,
)
from whisper_stream.projects.faster_whisper_api.launcher import launch_app

common_container = CommonServiceContainer()
model_container = FasterWhisperModelServiceContainer()

common_container.init_resources()

app: FastAPI = create_app(
    api_logger=common_container.logger(),
    app_metadata=common_container.app_metadata(),
    launch_config=common_container.launch_config(),
    common_container=common_container,
    model_container=model_container,
)


def main() -> None:
    launch_app("whisper_stream.projects.faster_whisper_api.__main__:app")


if __name__ == "__main__":
    main()
    common_container.shutdown_resources()
