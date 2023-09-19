from unittest import mock
from unittest.mock import patch
from marqo.client import Client
from tests.marqo_test import MarqoTestCase
import requests
from marqo.errors import BackendTimeoutError, BackendCommunicationError, BadRequestError, MarqoApiError


class TestHealth(MarqoTestCase):


    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def test_error_handling_in_health_check(self):
        client = Client(**self.client_settings)
        side_effect_list = [requests.exceptions.JSONDecodeError("test", "test", 1), BackendCommunicationError("test"),
                            BackendTimeoutError("test"), requests.exceptions.RequestException("test"),
                            KeyError("test"), KeyError("test"), requests.exceptions.Timeout("test")]
        for i, side_effect in enumerate(side_effect_list):
            with mock.patch("marqo._httprequests.HttpRequests.get") as mock_get:
                mock_get.side_effect = side_effect

                with self.assertRaises(BadRequestError) as cm:
                    res = client.health()

                # Assert the error message is what you expect
                self.assertIn("If you are trying to check the health on Marqo Cloud", cm.exception.message)

    def test_check_index_health_response(self):
        self.client.create_index(self.index_name_1)
        res = self.client.index(self.index_name_1).health()
        assert 'status' in res
        assert 'status' in res['backend']

    def test_check_index_health_query(self):
        with patch("marqo._httprequests.HttpRequests.get") as mock_get:
            self.client.create_index(self.index_name_1)
            res = self.client.index(self.index_name_1).health()
            args, kwargs = mock_get.call_args
            self.assertIn(f"/{self.index_name_1}/health", kwargs["path"])
