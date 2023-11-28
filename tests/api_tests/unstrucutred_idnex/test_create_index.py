from marqo.client import Client
from marqo import errors
from tests.marqo_test import MarqoTestCase


class CreateUnstructuredIndex(MarqoTestCase):
    UNSTRUCTURED = "unstructured"
    def setUp(self):
        self.client = Client(**self.client_settings)

    def test_illegal_index_name(self):
        with self.assertRaises(errors.MarqoWebError) as e:
            self.client.create_index("test-1", type=self.UNSTRUCTURED)

        self.assertIn("invalid_index_name", str(e.exception.message))
