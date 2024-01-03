import pytest
from marqo import errors
from marqo.client import Client

from tests.marqo_test import MarqoTestCase


@pytest.mark.fixed
class CreateUnstructuredIndex(MarqoTestCase):
    UNSTRUCTURED = "unstructured"
    def setUp(self):
        self.client = Client(**self.client_settings)

    def test_illegal_index_name(self):
        with self.assertRaises(errors.MarqoWebError) as e:
            self.client.create_index("test-1", type=self.UNSTRUCTURED)

        self.assertIn("not a valid index name", str(e.exception.message))
