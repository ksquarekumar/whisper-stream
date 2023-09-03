#!python3
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
from pathlib import Path
from typing import Any, Callable, Literal

import boto3
import click
from botocore.exceptions import NoCredentialsError
from joblib import Parallel, cpu_count, delayed

from whisper_stream.core.constants import DEFAULT_DATA_PATH
from whisper_stream.core.logger import BoundLogger, get_application_logger

from whisper_stream.core.helpers.data_loading import load_data_samples_from_path


def _invalidate_renaming_if_exists(filenames: list[str], directory: Path) -> None:
    found: list[str] = load_data_samples_from_path(
        filenames, directory, return_all=True
    )
    if len(found) > 0:
        error: str = f"Files exist, cannot rename"
        exception = FileExistsError(error)
        exception.add_note(f"The following files exist already:\n{', '.join(found)}")


def download_files_from_s3_and_rename(
    s3_client: Any,
    s3_bucket: str,
    s3_bucket_prefix: str,
    pattern: str,
    logger: BoundLogger,
    local_directory: Path = DEFAULT_DATA_PATH,
    rename_all: bool = True,
    rename_with: str = "audio",
) -> None:
    """Download files from S3 and optionally rename them by Index.

    Args:
        s3_client (boto3.client) an instance of boto3.client("s3")
        s3_bucket (str): The name of the S3 bucket.
        s3_bucket_prefix (str): The name of the S3 bucket prefix.
        pattern (str): The pattern to match files in the bucket.
        local_directory (Path, optional): The local directory to download files to. Defaults to DL_PATH.
        rename_all (bool, optional): Whether to rename all files by `Audio_{id}.mp3` or download them as-is. Defaults to True.
        rename_with (str): The prefix to use for renaming downloaded files

    Raises:
        EnvironmentError: S3 Client could not connect
        RuntimeError: Unhandled Exceptions
    """
    rename_fn: Callable[[str | int, str], str] = lambda i, ext: f"Audio_{i}.{ext}"
    # Initialize a Boto3 S3 client
    try:
        # List objects in the specified S3 bucket
        logger.info(
            "Listing objects in the S3 bucket",
            Bucket=s3_bucket,
            Prefix=s3_bucket_prefix,
        )
        objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_bucket_prefix)

        # Download files that match the pattern
        logger.info(
            "Downloading files from S3 bucket",
            Bucket=s3_bucket,
            Prefix=s3_bucket_prefix,
        )

        # Make jobs
        jobs: list[tuple[str, str, str, int]] = []
        filenames: list[str] = []
        idx: int = 1
        for obj in objects.get("Contents", []):
            if pattern in obj["Key"]:
                extension: str = obj["Key"].split(".")[-1]
                file_name: str = rename_fn(idx, extension) if rename_all else obj["Key"]
                local_path: str = str((local_directory / file_name).absolute())
                filenames.append(file_name)
                jobs.append((s3_bucket, str(obj["Key"]), local_path, obj["Size"]))
                idx += 1
        # sort by Size
        jobs = sorted(jobs, key=lambda x: x[3], reverse=True)
        # check if safe
        _invalidate_renaming_if_exists(filenames=filenames, directory=local_directory)
        # Download in parallel
        Parallel(n_jobs=cpu_count(), parallel_backend="threading")(
            delayed(s3_client.download_file)(bucket, obj_key, local_path)
            for bucket, obj_key, local_path, _ in jobs
        )

    except NoCredentialsError as e:
        logger.error("Credentials not available.")
        raise EnvironmentError(e) from e
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise RuntimeError(e) from e


@click.command()
@click.option(
    "--s3-bucket",
    required=True,
    prompt="S3 Bucket Name",
    help="Name of the S3 bucket",
    show_default=True,
    type=str,
)
@click.option(
    "--s3-bucket-prefix",
    required=True,
    prompt="S3 Bucket Prefix",
    help="Prefix of the S3 bucket",
    show_default=True,
    type=str,
)
@click.option(
    "--pattern",
    default=".mp3",
    prompt="File Pattern",
    help="Pattern to match files in the bucket",
    show_default=True,
    type=str,
)
@click.option(
    "--local-directory",
    default=DEFAULT_DATA_PATH,
    prompt="Local Directory",
    help="Local directory to save downloaded files",
    show_default=True,
    type=click.Path(),
)
@click.option(
    "--rename-all",
    default="No",
    prompt="Do you want to bulk rename ?",
    help="Whether to rename all files or download them as is",
    show_default=True,
    type=click.Choice(["Y", "N"]),
)
def download_files_from_s3_and_rename_cli(
    s3_bucket: str,
    s3_bucket_prefix: str,
    pattern: str,
    local_directory: Path = DEFAULT_DATA_PATH,
    rename_all: Literal["Y", "N"] = "N",
    rename_with: str = "audio",
) -> None:
    if rename_all == "Y":
        rename_with = click.prompt(
            "Enter prefix to rename files with",
            default="audio",
            show_default=True,
            hide_input=False,
            type=str,
        )

    click.confirm(
        f"You have selected to download in bulk all files matching *{pattern} from s3://{s3_bucket}/{s3_bucket_prefix} and rename them by {rename_with}.{'{.ext}'} in {local_directory}, does this seem alright?",
        abort=True,
    )

    logger: BoundLogger = get_application_logger(
        name="prefetch",
        binds={
            "fn": "download_files_from_s3_and_rename",
            "bucket": s3_bucket,
            "prefix": s3_bucket_prefix,
            "pattern": pattern,
            "local_directory": local_directory,
        },
    )

    s3_client = boto3.client("s3")

    download_files_from_s3_and_rename(
        s3_client=s3_client,
        s3_bucket=s3_bucket,
        s3_bucket_prefix=s3_bucket_prefix,
        logger=logger,
        pattern=pattern,
        local_directory=local_directory,
        rename_all=True if rename_all == "Y" else False,
        rename_with=rename_with,
    )


__all__: list[str] = ["download_files_from_s3_and_rename"]


if __name__ == "__main__":
    download_files_from_s3_and_rename_cli()
