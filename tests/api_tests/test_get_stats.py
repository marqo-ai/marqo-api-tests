from tests.marqo_test import MarqoTestCase
from marqo.errors import IndexNotFoundError
from marqo.client import Client

class TestGetStats(MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.generic_header = {'Content-type': 'application/json'}
        self.index_name = 'my-test-index-1'
        try:
            self.client.delete_index(self.index_name)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name)
        except IndexNotFoundError as s:
            pass

    def test_get_status_response_format(self):
        self.client.create_index(self.index_name)
        res = self.client.index(self.index_name).get_stats()
