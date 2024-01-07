import uuid

import pytest
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


@pytest.mark.cuda_test
@pytest.mark.fixed
class TestModlCacheManagement(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_index_name = "structured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text"},
                ],
                "tensorFields": ["title"]
            },
            {
                "indexName": cls.unstructured_index_name,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "unstructured",
            }
        ])

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    def test_get_cuda_info(self) -> None:
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                r = self.client.index(index_name).get_cuda_info()
                self.assertIn("cuda_devices", r)

    def test_get_cpu_info(self) -> None:
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                r = self.client.index(index_name).get_cpu_info()
                self.assertIn("cpu_usage_percent", r)
                self.assertIn("memory_used_percent", r)
                self.assertIn("memory_used_gb", r)

    def test_get_loaded_models_format(self) -> None:
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                r = self.client.index(index_name).get_loaded_models()
                self.assertIn("models", r)

    def test_eject_no_cached_model(self) -> None:
        # test eject a model that is NOT cached
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(index_name).eject_model("void_model", "void_device")
                self.assertIn("model_not_in_cache", str(e.exception.message))

    def test_eject_model(self) -> None:
        # test eject a model that is cached
        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                # Do a search to ensure the model is cached
                r = self.client.index(index_name).search("q", device="cuda")
                res = self.client.index(index_name).eject_model("sentence-transformers/all-MiniLM-L6-v2", "cuda")
                self.assertIn("successfully eject", str(res))





