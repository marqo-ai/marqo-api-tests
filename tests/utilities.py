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


def allow_environments(function, data):
    # TODO: Document
    def wrapper(*args, **kwargs):
        if os.environ["TESTING_CONFIGURATION"] not in data['allowed_environments']:
            return
        else:
            result = function(*args, **kwargs)
            return result
    return wrapper


def classwide_decorate(decorator, **kwargs):
    # TODO: Document
    def decorate(cls):
        for method in dir(cls):
            if not method.startswith('__'):
                setattr(cls, method, decorator(getattr(cls, method), kwargs))
        return cls
    return decorate