from tests import marqo_test
from tests import utilities
from marqo import Client
from marqo.errors import MarqoApiError, BackendCommunicationError, MarqoWebError
from tests import marqo_test


class TestLogOutPut(marqo_test.MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        # Ensures that marqo goes back to default state after these tests
        utilities.rerun_marqo_with_default_config(
            calling_class=cls.__name__
        )
        print("Marqo has been rerun with default env vars!")

    def test_log_output_default(self):
        """Marqo should default to info level"""
        process_object = utilities.run_marqo_process_with_env_vars(
            env_vars=["-e", f"MARQO_LOG_LEVEL=info"],
            calling_class=self.__class__.__name__
        )
        print("test_log_output_defaulttest_log_output_defaulttest_log_output_defaulttest_log_output_default")
        for l in process_object:
            print(l)

