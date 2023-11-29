from marqo.client import Client
from marqo import errors
from tests.marqo_test import MarqoTestCase


class CreateStructuredIndex(MarqoTestCase):
    STRUCTURED = "structured"
    def setUp(self):
        self.client = Client(**self.client_settings)

    def test_illegal_index_name(self):
        with self.assertRaises(errors.MarqoWebError) as e:
            self.client.create_index("test-1", type=self.STRUCTURED,
                                     all_fields=[{"name": "title", "type": "text"}],
                                     tensor_fields=["title"])

        self.assertIn("invalid_index_name", str(e.exception.message))
