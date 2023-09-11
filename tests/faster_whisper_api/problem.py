from functools import lru_cache
from pathlib import Path
from typing import Any

from requests import post, Response
from whisper_stream.core.helpers.data_loading import load_data_samples_from_path

DEFAULT_DATA_PATH: Path = Path(__file__) / "../../data"


class TranscriptionAPIEndpoint:
    def __init__(
        self,
        host: str,
        port: int,
        prefix: str | None = None,
        path: str = "/transcribe/audio",
    ) -> None:
        self._host: str = host
        self._port: int = port
        self._prefix: str | None = prefix
        self._path: str = path

    def _baseurl(self) -> str:
        if self._port == 80:
            return f'http://{self._host}'
        else:
            return f'http://{self._host}:{self._port}'

    def _route_address(self) -> str:
        return (
            f"{self._prefix}/{self._path}/"
            if self._prefix is not None
            else f"{self._path}/"
        )

    @property
    def base_url(self) -> str:
        return self._baseurl()

    @property
    def task_url(self) -> str:
        return self._route_address()


class TranscriptionProblem:
    def __init__(
        self,
        host: str,
        port: int,
        prefix: str | None = None,
        path: str = "/transcribe/audio",
        sample_glob: str = "audio_2.mp3",
        data_path: Path = DEFAULT_DATA_PATH,
    ) -> None:
        self._data_path: Path = data_path
        self._sample_glob: str = sample_glob
        self.endpoint = TranscriptionAPIEndpoint(
            host=host, 
            port=port, 
            prefix=prefix, 
            path=path
        )

    @lru_cache
    def prepare_data(self) -> None:
        self.data: bytes = load_data_samples_from_path(
            sample_file_glob_or_globs=self._sample_glob,
            directory=self._data_path,
            return_all=False,
            binary_mode=True,
        )

    def run_problem(self) -> Any:
        self.prepare_data()
        response: Response = post(
            url=self._endpoint.task_url, files=dict(audio=self.data)
        )
        return response.json()
