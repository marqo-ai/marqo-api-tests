import pytest
import os

def pytest_configure(config):
    config.addinivalue_line("markers", "cuda_test: mark test as cuda_test to skip")
    config.addinivalue_line("markers", "non_cuda_test: mark test as non_cuda_test to skip")


def pytest_collection_modifyitems(items):
    if os.environ["TESTING_CONFIGURATION"] in ["CUDA_DIND_MARQO_OS"]:
        # Skip non-cuda-tests if the env is CUDA
        skip_non_cuda_test = pytest.mark.skip(reason="need to not set "
                                                     "'TESTING_CONFIGURATION=CUDA_DIND_MARQO_OS' to run")
        for item in items:
            if "non_cuda_test" in item.keywords:
                item.add_marker(skip_non_cuda_test)
    else:
        # Skip cuda-tests if the env is NOT CUDA
        skip_cuda_test = pytest.mark.skip(reason="need to setEnvVars "
                                                 "'TESTING_CONFIGURATION=CUDA_DIND_MARQO_OS' to run")
        for item in items:
            if "cuda_test" in item.keywords:
                item.add_marker(skip_cuda_test)