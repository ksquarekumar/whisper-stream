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
import fastapi
from fastapi import FastAPI

from whisper_stream.projects.faster_whisper_api.di.common import (
    AppMetaData,
    CommonServiceContainer,
)
from whisper_stream.projects.faster_whisper_api.di.faster_whisper_model import (
    FasterWhisperModelServiceContainer,
)
from whisper_stream.projects.faster_whisper_api.schemas.exceptions import (
    api_exception_handlers,
)
from whisper_stream.projects.faster_whisper_api.endpoints import (
    probes_router,
    transcription_router,
)
from whisper_stream.projects.faster_whisper_api.logger import APIBoundLogger

# di containers
common_container = CommonServiceContainer()
model_container = FasterWhisperModelServiceContainer()


# app factory
def create_app(api_logger: APIBoundLogger = common_container.logger()) -> FastAPI:
    app_metadata: AppMetaData = common_container.app_metadata()
    app: FastAPI = FastAPI(
        title=app_metadata.full_name,
        description=app_metadata.description,
        version=app_metadata.version,
        docs_url="/openapi",
        redoc_url="/docs",
        exception_handlers=api_exception_handlers,
        default_response_class=fastapi.responses.ORJSONResponse,
    )
    app.logger = api_logger  # type: ignore[attr-defined]
    app.configs_container = common_container  # type: ignore[attr-defined]
    app.model_container = model_container  # type: ignore[attr-defined]
    app.include_router(probes_router)
    app.include_router(transcription_router)
    return app


__all__: list[str] = ["create_app", "common_container", "model_container"]
