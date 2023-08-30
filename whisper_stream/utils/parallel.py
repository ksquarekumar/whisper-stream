# SPDX-FileCopyrightText: 2023-present krishnakumar <krishna.kumar@peak.ai>
#
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
import pathlib
from typing import Literal
from joblib import Parallel, cpu_count, delayed

ParallelBackendTypes = Literal["loky", "threading", "multiprocessing"]


@lru_cache(typed=True)  # type: ignore[misc]
def get_backend(
    backend: ParallelBackendTypes = "loky",
    n_jobs: int | None = None,
    count_only_physical_cores: bool = True,
    batch_size: str | int = "auto",
    require: Literal["sharedmem", None] = None,
    temp_folder: str | None = None,
    mmap_mode: Literal[None, 'r+', 'r', 'w+', 'c'] = None,
    max_nbytes: str | int | None = None,
) -> Parallel:
    """Helper function for `Joblib`'s readable parallel mapping.

    Args:
        backend (ParallelBackendTypes):
            Specify the parallelization backend implementation, defaults to `loky`

            Supported backends are:
            - "loky" used by default, can induce some
              communication and memory overhead when exchanging input and
              output data with the worker Python processes. On some rare
              systems (such as Pyiodide), the loky backend may not be
              available.
            - "multiprocessing" previous process-based backend based on
              `multiprocessing.Pool`. Less robust than `loky`.
            - "threading" is a very low-overhead backend but it suffers
              from the Python Global Interpreter Lock if the called function
              relies a lot on Python objects. "threading" is mostly useful
              when the execution bottleneck is a compiled extension that
              explicitly releases the GIL (for instance a Cython loop wrapped
              in a "with nogil" block or an expensive call to a library such
              as NumPy).
        n_jobs (int | None, optional): _
            The maximum number of concurrently running jobs, Defaults to None.

            If -1 all CPUs are used, if 1 is given, no parallel computing code is used at all
            None effectively sets `n_jobs` to -1 through `joblib.cpu_count`
        count_only_physical_cores (bool, optional):
            Used to calculate the number of available cores, Defaults to None.

            Ignored when `n_jobs` is set.
            When only_physical_cores is True,`joblib.cpu_count` does not take hyperthreading / SMT logical cores into account
        batch_size (str | int, optional):
            The number of atomic tasks to dispatch at once. Defaults to "auto".

            When individual evaluations are very fast, dispatching
            calls to workers can be slower than sequential computation because
            of the overhead. Batching fast computations together can mitigate
            this.
            The ``'auto'`` strategy keeps track of the time it takes for a
            batch to complete, and dynamically adjusts the batch size to keep
            the time on the order of half a second, using a heuristic. The
            initial batch size is 1.
            ``batch_size="auto"`` with ``backend="threading"`` will dispatch
            batches of a single task at a time as the threading backend has
            very little overhead and using larger batch size has not proved to
            bring any gain in that case.
        require (Literal[&quot;sharedmem&quot;, None], optional):
            Hard constraint to select the backend, Defaults to None..
            If set to 'sharedmem', the selected backend will be single-host
            and thread-based even if the user asked for a non-thread based backend with
            :func:`~joblib.parallel_config`
        temp_folder (str | None, optional):
            Folder to be used by the pool for memmapping large arrays
            for sharing memory with worker processes, Defaults to None.

            If None, this will try in order:
            - a folder pointed by the JOBLIB_TEMP_FOLDER environment
              variable,
            - /dev/shm if the folder exists and is writable: this is a
              RAM disk filesystem available by default on modern Linux
              distributions,
            - the default system temporary folder that can be
              overridden with TMP, TMPDIR or TEMP environment
              variables, typically /tmp under Unix operating systems.

        mmap_mode (Literal[None, &#39;r, optional):
            Memmapping mode for numpy arrays passed to workers, Defaults to None.
            None will disable memmapping, other modes defined in the numpy.memmap doc:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
            Also, see 'max_nbytes' parameter documentation for more details.

        max_nbytes (str | int | None, optional):
            Threshold on the size of arrays passed to the workers that
            triggers automated memory mapping in temp_folder, Defaults to None.
            Can be an int in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
            Use None to disable memmapping of large arrays.
            Only active when backend="loky" or "multiprocessing".

    Raises:
        err: _description_
        ValueError: _description_

    Returns:
        Parallel: _description_
    """

    _n_jobs: int = n_jobs if n_jobs is not None else (cpu_count(count_only_physical_cores) or -1)

    if temp_folder is not None:
        try:
            pathlib.Path(temp_folder).mkdir(exist_ok=True)
            temp_folder = str(pathlib.Path(temp_folder).absolute())
        except Exception as err:
            raise err

    match backend:
        case "threading":
            return Parallel(
                backend=backend,
                n_jobs=_n_jobs,
                batch_size=str(batch_size),
                require=require,
                temp_folder=temp_folder,
                mmap_mode=mmap_mode,
            )
        case "loky" | "multiprocessing":
            return Parallel(
                backend=backend,
                n_jobs=_n_jobs,
                batch_size=str(batch_size),
                require=require,
                temp_folder=temp_folder,
                max_nbytes=max_nbytes,
            )
        case _:
            error: str = (f"Backend: {backend} does not exist(!)",)  # type: ignore[unreachable]
            raise ValueError(error)


DEFAULT_PARALLEL_BACKEND: Parallel = get_backend("threading")

__all__: list[str] = ["get_backend", "cpu_count", "delayed", "DEFAULT_PARALLEL_BACKEND"]
