import os
import subprocess
import typing
import threading
import inspect


def disallow_environments(disallowed_configurations: typing.List[str]):
    """This construct wraps a test to ensure that it does not run for disallowed 
    testing environments.

    It figures by examining the "TESTING_CONFIGURATION" environment variable.

    Args:
        disallowed_configurations: if the environment variable
        "TESTING_CONFIGURATION" matches a configuration in
        disallowed_configurations, then the test will be skipped
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            if os.environ["TESTING_CONFIGURATION"] in disallowed_configurations:
                return
            else:
                result = function(*args, **kwargs)
                return result
        return wrapper
    return decorator


def allow_environments(allowed_configurations: typing.List[str]):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if os.environ["TESTING_CONFIGURATION"] not in allowed_configurations:
                return
            else:
                result = function(*args, **kwargs)
                return result
        return wrapper
    return decorator


def classwide_decorate(decorator, allowed_configurations):
    def decorate(cls):
        for method in dir(cls):
            if method.startswith("test"):
                setattr(cls, method, (decorator(allowed_configurations))(getattr(cls, method)))
        return cls
    return decorate


def rerun_marqo_with_env_vars(env_vars: list = [], calling_class: str = ""):
    """
        Given a list of env vars / flags, stop and rerun Marqo using the start script appropriate
        for the current test config

        Ensure that:
        1. Flags are separate items from variable itself (eg, ['-e', 'MARQO_MODELS_TO_PRELOAD=["hf/all_datasets_v4_MiniLM-L6"]'])
        2. Strings (individual items in env_vars list) do not contain ' (use " instead)
        -> single quotes cause some parsing issues and will affect the test outcome
    """

    if calling_class not in ["TestEnvVarChanges"]:
        raise RuntimeError(
            f"Rerun Marqo function should only be called by `TestEnvVarChanges` to ensure other API tests are not affected. Given calling class is {calling_class}")

    # Stop Marqo
    print("Attempting to stop marqo.")
    subprocess.run(["docker", "stop", "marqo"], check=True, capture_output=True)
    print("Marqo stopped.")

    # Rerun the appropriate start script
    test_config = os.environ["TESTING_CONFIGURATION"]

    if test_config == "LOCAL_MARQO_OS":
        start_script_name = "start_local_marqo_os.sh"
    elif test_config == "DIND_MARQO_OS":
        start_script_name = "start_dind_marqo_os.sh"
    elif test_config == "S2SEARCH_OS":
        start_script_name = "start_s2search_backend.sh"
    elif test_config == "ARM64_LOCAL_MARQO_OS":
        start_script_name = "start_arm64_local_marqo_os.sh"
    elif test_config == "CUDA_DIND_MARQO_OS":
        start_script_name = "start_cuda_dind_marqo_os.sh"
    full_script_path = f"{os.environ['MARQO_API_TESTS_ROOT']}/scripts/{start_script_name}"

    run_process = subprocess.Popen(
        [
            "bash",  # command: run
            full_script_path,  # script to run
            os.environ['MARQO_IMAGE_NAME'],  # arg $1 in script
        ] + env_vars,  # args $2 onwards
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    lines = []
    # Read and print the output line by line (in real time)
    for line in run_process.stdout:
        lines.append(line)

    # Wait for the process to complete
    run_process.wait()
    return lines


def rerun_marqo_with_default_config(calling_class: str = ""):
    # Do not send any env vars
    # This should act like running the start script at the beginning
    rerun_marqo_with_env_vars(env_vars=[], calling_class=calling_class)


docker_log_failure_message = "Failed to fetch docker logs for Marqo"


def fetch_docker_logs(container_name: str, log_collection: typing.List):
    """Fetches the Docker logs of a specified container and stores them in a provided list.

    Args:
        container_name (str): Name of the Docker container whose logs are to be fetched.
        log_collection (List): A list where the fetched logs or error messages are stored.
    """
    completed_process = subprocess.run(
        ["docker", "logs", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if completed_process.returncode == 0:
        log_collection.append(completed_process.stdout)
    else:
        log_collection.append(
            f"{docker_log_failure_message}. "
            f"Failed with error: {completed_process.stderr}")


def check_logs(
        log_wide_checks: typing.List[typing.Callable[[str], bool]],
        container_name: str
):
    """Checks the logs of a specified Docker container for the presence or absence of specific lines.

    Args:
        log_wide_checks (List[str]): a list of functions to be applied to each to the entire log sr. These
            functions should test for some property, such as the presence or absence of a substring, returning
            True iff it passes
        container_name (str): Name of the Docker container whose logs are to be checked. Defaults to 'marqo'.
    Raises:
        RuntimeError: If fetching logs fails or times out.
    """
    # a 1-elem mutable object to save the docker logs to:
    log_collection = []

    docker_log_fetcher = fetch_docker_logs

    # Run the fetch_docker_logs function in a separate thread
    thread = threading.Thread(target=docker_log_fetcher, args=(container_name, log_collection))
    thread.start()
    thread.join(timeout=30)

    if not log_collection:
        raise RuntimeError("Fetching logs timed out or failed. log_collection is empty.")

    if docker_log_failure_message in log_collection[0]:
        raise RuntimeError(f"{docker_log_fetcher.__name__} encountered an error retrieving docker logs. {log_collection[0]}")

    return log_collection[0]