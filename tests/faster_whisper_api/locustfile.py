from typing import Final
from locust import TaskSet, task, between, FastHttpUser
from faster_whisper_api.problem import TranscriptionProblem


API_HOST: Final[str] = "localhost"
API_PORT: Final[int] = 8080
API_PREFIX: Final[str | None] = None
API_PATH: Final[str] = "transcribe/audio"


TranscriptionLocust = TranscriptionProblem(
    host=API_HOST,
    port=API_PORT,
    prefix=API_PREFIX,
    path=API_PATH
)


class LoadTasks(TaskSet):

    @task
    def post(self) -> None:
        url: str = TranscriptionLocust.endpoint.task_url
        self.client.post(url, files=dict(audio=TranscriptionLocust.data))

class EndpointUser(FastHttpUser):
    """
    User class that does requests to the locust web server running on localhost
    """
    host = TranscriptionLocust.endpoint.task_url
    wait_time = between(1,1)
    tasks = [LoadTasks]