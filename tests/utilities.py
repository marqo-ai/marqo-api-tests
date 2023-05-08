import os
import typing


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


def rerun_marqo_with_env_vars(env_vars: str = ""):
    """
        Given a string of env vars, stop and rerun Marqo using the start script appropriate
        for the current test config
    """
    # Stop Marqo
    subprocess.run(["docker", "stop", "marqo"], check=True, capture_output=True)

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
    
    subprocess.run([
        ".",                                # command: run
        full_script_path,                   # script to run
        os.environ['MARQO_IMAGE_NAME'],     # arg $1 in script
        env_vars                            # arg $2 in script
        ], check=True, capture_output=True)