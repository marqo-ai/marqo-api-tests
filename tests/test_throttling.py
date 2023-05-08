import copy
import pprint
from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError, MarqoWebError
import unittest
from tests.marqo_test import MarqoTestCase
from marqo import enums
from unittest import mock


class TestThrottling(MarqoTestCase):

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
        
    """
    def test_max_throttle(self):
        self.client.create_index(index_name=self.index_name_1)
        pass
    """