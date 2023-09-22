from tests import marqo_test
from tests import utilities
from marqo import Client
from marqo.errors import MarqoApiError, BackendCommunicationError, MarqoWebError
import pprint
import json
from tests.application_tests.test_env_var_changes import  TestEnvVarChanges


class TestLogOutPut(TestEnvVarChanges):

    def test_log_output_default(self):
        """Marqo should default to info level"""
        process_object = utilities.run_marqo_process_with_env_vars(
            env_vars=["-e", f"MARQO_LOG_LEVEL=INFO"],
            calling_class=self.__class__.__name__
        )
        for l in process_object:
            print(l)

