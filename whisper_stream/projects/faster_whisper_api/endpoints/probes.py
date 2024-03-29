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
from datetime import datetime
from dependency_injector.wiring import Provide, inject
from whisper_stream.projects.faster_whisper_api.di.common import (
    AppMetaData,
    CommonServiceContainer,
)
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, RedirectResponse

probes_router = APIRouter()


@probes_router.get("/healthz", operation_id="healthz")
@inject
async def healthz(
    app_metadata: AppMetaData = Depends(Provide[CommonServiceContainer.app_metadata]),
) -> PlainTextResponse:
    return PlainTextResponse(
        f"you have arrived at {app_metadata.full_name} {datetime.utcnow().isoformat()}",
        200,
    )


@probes_router.get(
    "/ping",
    operation_id="ping",
)
@inject
async def ping(
    app_metadata: AppMetaData = Depends(Provide[CommonServiceContainer.app_metadata]),
) -> PlainTextResponse:
    return PlainTextResponse(
        f"pong from {app_metadata.full_name} at {datetime.utcnow().isoformat()}", 200
    )


@probes_router.get("/", operation_id="redirect-to-openapi", include_in_schema=False)
async def get_redirect() -> RedirectResponse:
    return RedirectResponse(url="/openapi")


__all__: list[str] = ["probes_router"]
