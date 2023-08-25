#!python3
from typing import Callable, Literal, overload
import boto3
from botocore.exceptions import NoCredentialsError
from pathlib import Path
from joblib import Parallel, delayed, cpu_count

import click
from whisper_stream.logger import get_application_logger
import structlog

DL_PATH: Path = Path(__file__).parent.absolute()


@overload
def load_data_sample_from_path(sample_file: str, binary_mode: Literal[False] = False) -> str:
    ...


@overload
def load_data_sample_from_path(sample_file: str, binary_mode: Literal[True]) -> bytes:
    ...


def load_data_sample_from_path(sample_file: str, binary_mode: Literal[True, False] = False) -> str | bytes:
    files = list(DL_PATH.glob(f"{sample_file}"))
    if len(files) >= 1:
        if binary_mode == False:
            return str(files[0].absolute())
        with files[0].absolute():
            return files[0].read_bytes()
    msg = f"File {sample_file} does not exist in ./data"
    raise FileNotFoundError(msg)


@click.command()
@click.option('--bucket', required=True, prompt='S3 Bucket Name', help='Name of the S3 bucket', show_default=True)
@click.option('--prefix', required=True, prompt='S3 Bucket Prefix', help='Prefix of the S3 bucket', show_default=True)
@click.option(
    '--pattern', default=".mp3", prompt='File Pattern', help='Pattern to match files in the bucket', show_default=True
)
@click.option(
    '--local-directory',
    default=DL_PATH,
    prompt='Local Directory',
    help='Local directory to save downloaded files',
    show_default=True,
    type=click.Path(),
)
@click.option(
    '--rename-all',
    default=True,
    is_flag=True,
    help='Whether to rename all files or download them as is',
    show_default=True,
)
def download_files_from_s3_and_rename(
    bucket: str, prefix: str, pattern: str, local_directory: Path = DL_PATH, rename_all: bool = True
) -> None:
    """Download files from S3 and optionally rename them by Index.

    Args:
        bucket (str): The name of the S3 bucket.
        prefix (str): The name of the S3 bucket prefix.
        pattern (str): The pattern to match files in the bucket.
        local_directory (Path, optional): The local directory to download files to. Defaults to DL_PATH.
        rename_all (bool, optional): Whether to rename all files by `Audio_{id}.mp3` or download them as-is. Defaults to True.

    Raises:
        EnvironmentError: S3 Client could not connect
        RuntimeError: Unhandled Exceptions
    """
    rename_fn: Callable[[str | int], str] = lambda i: f"Audio_{i}.mp3"
    # Initialize a Boto3 S3 client
    logger: structlog.stdlib.BoundLogger = get_application_logger(
        name="prefetch",
        binds={
            "fn": "download_files_from_s3_and_rename",
            "bucket": bucket,
            "prefix": prefix,
            "pattern": pattern,
            "local_directory": local_directory,
        },
    )
    try:
        # List objects in the specified S3 bucket
        logger.info("Listing objects in the S3 bucket", Bucket=bucket, Prefix=prefix)
        objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        # Download files that match the pattern
        logger.info("Downloading files from S3 bucket", Bucket=bucket, Prefix=prefix)

        # Make jobs
        jobs: list[tuple[str, str, str, int]] = []
        idx: int = 1
        for obj in objects.get("Contents", []):
            if pattern in obj["Key"]:
                file_name: str = rename_fn(idx) if rename_all else obj['Key']
                local_path: str = str((local_directory / file_name).absolute())
                jobs.append((bucket, str(obj['Key']), local_path, obj["Size"]))
                idx += 1
        # sort by Size
        jobs = sorted(jobs, key=lambda x: x[3], reverse=True)
        # Download in parallel
        Parallel(n_jobs=cpu_count(), backend="threading")(
            delayed(s3_client.download_file)(bucket, obj_key, local_path) for bucket, obj_key, local_path, _ in jobs
        )

    except NoCredentialsError as e:
        logger.error("Credentials not available.")
        raise EnvironmentError(e) from e
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise RuntimeError(e) from e


__all__: list[str] = ["load_data_sample_from_path"]

if __name__ == "__main__":
    s3_client: boto3.client = boto3.client('s3')
    download_files_from_s3_and_rename()
