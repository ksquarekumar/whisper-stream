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
from typing import Any, Callable, Coroutine
from fastapi import HTTPException, Request, Response
from fastapi.exception_handlers import websocket_request_validation_exception_handler
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.responses import ORJSONResponse
from pydantic import Field

from faster_whisper_api.config.base import APIBaseModel


class APIExceptionDetails(APIBaseModel):
    message: str
    extra: list[str] | None = None


class APIException(APIBaseModel):
    detail: APIExceptionDetails = Field(
        description="A human-readable explanation specific to this occurrence of the problem"
    )
    error: str = Field(description="Exception details (e.g. validation result)")
    status_code: int = Field(description="The HTTP status code")
    title: str = Field(
        description="A short, human-readable summary of the problem type"
    )
    type: str = Field(
        description="A URI reference [RFC3986] that identifies the problem type"
    )
    headers: None = None

    def __init__(self, error: Exception, detail: APIExceptionDetails) -> None:
        self.error = type(error).__name__
        self.detail: APIExceptionDetails = detail or {
            "message": str(error),
            "extra": error.__notes__ or None,
        }


class BadRequestAPIException(APIException):
    status_code: int = Field(default=400, description="The HTTP status code")
    title: str = Field(
        default="request was invalid.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.1",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class UnAuthorizedAPIException(APIException):
    status_code: int = Field(default=401, description="The HTTP status code")
    title: str = Field(
        default="unauthorized.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.2",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class ForbiddenAPIException(APIException):
    status_code: int = Field(default=403, description="The HTTP status code")
    title: str = Field(
        default="access to resource is forbidden.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.4",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class NotFoundAPIException(APIException):
    status_code: int = Field(default=404, description="The HTTP status code")
    title: str = Field(
        default="resource not found.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.5",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class PayloadExceededAPIException(APIException):
    status_code: int = Field(default=413, description="The HTTP status code")
    title: str = Field(
        default="content size too large.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.14",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class InvalidContentTypeExceptionRespone(APIException):
    status_code: int = Field(default=415, description="The HTTP status code")
    title: str = Field(
        default="content-type not supported for this operation.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.16",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class UnprocessableEntityAPIException(APIException):
    status_code: int = Field(default=422, description="The HTTP status code")
    title: str = Field(
        default="Unable to process request.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.21",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class InternalServerAPIException(APIException):
    status_code: int = Field(default=500, description="The HTTP status code")
    title: str = Field(
        default="An error occurred while processing your request.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.6.4",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


class ServiceUnavailableAPIException(APIException):
    status_code: int = Field(default=503, description="The HTTP status code")
    title: str = Field(
        default="Service is temporarily unavailable.",
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default="https://www.rfc-editor.org/rfc/rfc9110.html#section-15.6.1",
        description="A URI reference [RFC3986] that identifies the problem type",
    )


async def http_exception_handler(
    _, exc: APIException | HTTPException
) -> ORJSONResponse:
    content: dict[str, Any] = {
        "error": exc.error or "unknown",
        "detail": exc.detail.model_dump_json(exclude_none=True)
        if isinstance(exc, APIBaseModel)
        else exc.detail,
        "title": exc.title or "unknown",
        "type": exc.type or "unknown",
    }
    return ORJSONResponse(content, status_code=exc.status_code)


async def validation_exception_handler(
    request: Any,
    exc: BadRequestAPIException
    | UnprocessableEntityAPIException
    | RequestValidationError,
) -> ORJSONResponse:
    content: dict[str, Any] = {
        "error": exc.error
        if isinstance(exc.error, str)
        else [str(err) for err in list(exc.errors)],
        "detail": exc.detail.model_dump_json(exclude_none=True)
        if isinstance(exc, APIBaseModel)
        else (exc.body or {}),
        "title": exc.title or "Failed to process request.",
        "type": exc.type
        or "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.21",
    }
    return ORJSONResponse(content, status_code=int(exc.status_code or 422))


api_exception_handlers: dict[
    int | type[Exception],
    Callable[[Request, Any], Coroutine[Any, Any, Response] | None],
] = {
    RequestValidationError: validation_exception_handler,
    HTTPException: http_exception_handler,
    WebSocketRequestValidationError: websocket_request_validation_exception_handler,  # type: ignore
}
