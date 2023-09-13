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
from dataclasses import dataclass


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


@dataclass(frozen=False)
class APIExceptionDetails:
    message: str
    extra: list[str] | None = None


@dataclass(frozen=False)
class APIException(BaseException):
    error: str
    detail: APIExceptionDetails
    status_code: int
    title: str
    type: str
    headers: None = None

    def __init__(
        self, error: Exception, detail: APIExceptionDetails | None = None
    ) -> None:
        self.error = type(error).__name__
        self.detail: APIExceptionDetails = (
            detail
            if detail is not None
            else APIExceptionDetails(
                message=str(error), extra=getattr(error, "__notes__", None)
            )
        )


class BadRequestAPIException(APIException):
    status_code: int = 400
    title: str = get_default_title(400)
    type: str = get_default_error_rfc_url(400)


class UnAuthorizedAPIException(APIException):
    status_code: int = 401
    title: str = get_default_title(401)
    type: str = get_default_error_rfc_url(401)


class ForbiddenAPIException(APIException):
    status_code: int = 403
    title: str = get_default_title(403)
    type: str = get_default_error_rfc_url(403)


class NotFoundAPIException(APIException):
    status_code: int = 404
    title: str = get_default_title(404)
    type: str = get_default_error_rfc_url(404)


class PayloadExceededAPIException(APIException):
    status_code: int = 413
    title: str = get_default_title(413)
    type: str = get_default_error_rfc_url(413)


class InvalidContentTypeExceptionRespone(APIException):
    status_code: int = 415
    title: str = get_default_title(415)
    type: str = get_default_error_rfc_url(415)


class UnprocessableEntityAPIException(APIException):
    status_code: int = 422
    title: str = get_default_title(422)
    type: str = get_default_error_rfc_url(422)


class InternalServerAPIException(APIException):
    status_code: int = 500
    title: str = get_default_title(500)
    type: str = get_default_error_rfc_url(500)


class ServiceUnavailableAPIException(APIException):
    status_code: int = 503
    title: str = get_default_title(503)
    type: str = get_default_error_rfc_url(503)
