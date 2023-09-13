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
from typing import Final
from locust import TaskSet, task, between, FastHttpUser
from faster_whisper_api.problem import TranscriptionProblem


API_HOST: Final[str] = "localhost"
API_PORT: Final[int] = 8080
API_PREFIX: Final[str | None] = None
API_PATH: Final[str] = "transcribe/audio"


TranscriptionLocust = TranscriptionProblem(
    host=API_HOST, port=API_PORT, prefix=API_PREFIX, path=API_PATH
)


class LoadTasks(TaskSet):
    @task
    def post(self) -> None:  # type: ignore[misc]
        url: str = TranscriptionLocust.endpoint.task_url
        self.client.post(url, files=dict(audio=TranscriptionLocust.data))


class EndpointUser(FastHttpUser):
    """
    User class that does requests to the locust web server running on localhost
    """

    host = TranscriptionLocust.endpoint.task_url
    wait_time = between(1, 1)  # type: ignore[no-untyped-call]
    tasks = [LoadTasks]
