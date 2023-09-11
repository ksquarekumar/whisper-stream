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
from pydantic import Field, dataclasses

from whisper_stream.projects.faster_whisper_api.config.base import APIBaseModel


def get_default_title(status_code: int) -> str:
    match status_code:
        case 400:
            return "request was invalid."
        case 401:
            return "unauthorized."
        case 403:
            return "access to resource is forbidden."
        case 404:
            return "resource not found."
        case 413:
            return "request entity too large."
        case 415:
            return "unsupported media type."
        case 422:
            return "request content invalid."
        case 500:
            return "server error."
        case 503:
            return "service unavailable."
        case _:
            return "unknown error."


def get_default_error_rfc_url(status_code: int) -> str:
    match status_code:
        case 400:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.1"
        case 401:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.2"
        case 403:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.4"
        case 404:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.5"
        case 413:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.14"
        case 415:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.16"
        case 422:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.21"
        case 500:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.6.1"
        case 503:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.6.4"
        case _:
            return "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.6"


class APIExceptionDetails(APIBaseModel):
    message: str
    extra: list[str] | None = None


@dataclasses.dataclass()
class APIException(BaseException):
    detail: APIExceptionDetails = Field(
        description="A human-readable explanation specific to this occurrence of the problem"
    )
    error: str = Field(description="Exception details (e.g. validation result)")
    status_code: int = Field(description="The HTTP status code")
    title: str = Field(
        description="A short, human-readable summary of the problem type"
    )
    type: str = Field(
        description="A URI reference [RFC9110] that identifies the problem type"
    )
    headers: None = None

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        self.error = type(error).__name__
        self.detail: APIExceptionDetails = (
            detail
            if detail is not None
            else APIExceptionDetails(message=str(error), extra=error.__notes__ or None)
        )
        super().__init__(
            self.error, self.detail, self.status_code, self.title, self.type
        )


class BadRequestAPIException(APIException):
    status_code: int = Field(default=400, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(400),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(400),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class UnAuthorizedAPIException(APIException):
    status_code: int = Field(default=401, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(401),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(401),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class ForbiddenAPIException(APIException):
    status_code: int = Field(default=403, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(403),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(403),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class NotFoundAPIException(APIException):
    status_code: int = Field(default=404, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(404),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(404),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class PayloadExceededAPIException(APIException):
    status_code: int = Field(default=413, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(413),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(413),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class InvalidContentTypeExceptionRespone(APIException):
    status_code: int = Field(default=415, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(415),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(415),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class UnprocessableEntityAPIException(APIException):
    status_code: int = Field(default=422, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(422),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(422),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class InternalServerAPIException(APIException):
    status_code: int = Field(default=500, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(500),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(500),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


class ServiceUnavailableAPIException(APIException):
    status_code: int = Field(default=503, description="The HTTP status code")
    title: str = Field(
        default=get_default_title(503),
        description="A short, human-readable summary of the problem type",
    )
    type: str = Field(
        default=get_default_error_rfc_url(503),
        description="A URI reference [RFC9110] that identifies the problem type",
    )

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        super().__init__(error=error, detail=detail)


async def http_exception_handler(
    request: Any, exc: APIException | HTTPException
) -> ORJSONResponse:
    content: dict[str, Any] = {
        "error": exc.error if isinstance(exc, APIException) else exc.detail,
        "detail": exc.detail.model_dump_json(exclude_none=True)
        if isinstance(exc, APIException)
        else exc.detail,
        "title": exc.title
        if isinstance(exc, APIException)
        else get_default_title(exc.status_code),
        "type": exc.type
        if isinstance(exc, APIException)
        else get_default_error_rfc_url(exc.status_code),
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
        if isinstance(exc, APIException)
        else [str(err) for err in list(exc.errors())],
        "detail": exc.detail.model_dump_json(exclude_none=True)
        if isinstance(exc, APIException)
        else (exc.body or {}),
        "title": exc.title if isinstance(exc, APIException) else get_default_title(422),
        "type": exc.type
        if isinstance(exc, APIException)
        else get_default_error_rfc_url(422)
        or "https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.21",
    }
    return ORJSONResponse(
        content,
        status_code=int(exc.status_code if isinstance(exc, APIException) else 422),
    )


api_exception_handlers: dict[
    int | type[Exception], Callable[[Request, Any], Coroutine[Any, Any, Response]]
] | None = {
    RequestValidationError: validation_exception_handler,
    HTTPException: http_exception_handler,
    WebSocketRequestValidationError: websocket_request_validation_exception_handler,  # type: ignore
}
