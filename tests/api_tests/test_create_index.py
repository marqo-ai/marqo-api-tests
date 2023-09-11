from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoWebError
import unittest
import pprint
from tests.marqo_test import MarqoTestCase


class TestCreateIndex(MarqoTestCase):

    def setUp(self) -> None:
        client_0 = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        for ix_name in ['some-bulk', self.index_name_1]:
            try:
                client_0.delete_index(ix_name)
            except MarqoApiError as s:
                pass

    def test_illegal_index_name_prevented(self):
        client = Client(**self.client_settings)

        client.create_index('some-bulk')
        try:
            client.create_index('bulk')
            raise AssertionError('created index with illegal name `bulk`!')
        except MarqoWebError:
            pass
        # ensure the index was not accidentally created despite the error:
        assert 'bulk' not in [ix.index_name for ix in client.get_indexes()['results']]
        # but an index name with 'bulk' as a substring should appear as expected:
        assert 'some-bulk' in [ix.index_name for ix in client.get_indexes()['results']]

    def tearDown(self) -> None:
        client_0 = Client(**self.client_settings)
        for ix_name in ['some-bulk', self.index_name_1]:
            try:
                client_0.delete_index(ix_name)
            except MarqoApiError as s:
                pass