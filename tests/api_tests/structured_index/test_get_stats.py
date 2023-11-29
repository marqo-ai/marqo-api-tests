import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase

class TestStructuredGetStats(MarqoTestCase):
    text_index_name = "api_test_structured_index" + str(uuid.uuid4()).replace('-', '')
    image_index_name = "api_test_structured_image_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "index_name": cls.text_index_name,
                "settings_dict": {
                    "type": "structured",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "all_fields": [
                        {"name": "title", "type": "text"},
                        {"name": "content", "type": "text"},
                    ],
                    "tensor_fields": ["title", "content"],
                }
            },
            {
                "index_name": cls.image_index_name,
                "settings_dict": {
                    "type": "structured",
                    "model": "open_clip/ViT-B-32/openai",
                    "all_fields": [
                        {"name": "title", "type": "text"},
                        {"name": "image_content", "type": "image_pointer"},
                    ],
                    "tensor_fields": ["title", "image_content"],
                }
            }
        ])

        cls.indexes_to_delete.extend([cls.text_index_name, cls.image_index_name])
    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)
            
    def test_get_status_response_format(self):
        res = self.client.index(self.text_index_name).get_stats()
        assert isinstance(res, dict)
        assert "numberOfVectors" in res
        assert "numberOfDocuments" in res

    def test_get_status_response_results(self):
        self.client.index(self.text_index_name).add_documents(
            documents=[
                {"title": "test-2", "content": "test"},  # 2 vectors
                {"title": "test-2", "content": "test", "description_3": "test"},  # 3 vectors
                {"description_2": "test"},  # 1 vector
                {"my_multi_modal_field": {
                    "text_1": "test", "text_2": "test"}},  # 1 vector
                {"non_tensor_field": "test"}  # 0 vectors
            ],
            auto_refresh=True, device="cpu",
            non_tensor_fields=["non_tensor_field"],
            mappings={"my_multi_modal_field": {"type": "multimodal_combination", "weights": {
                "text_1": 0.5, "text_2": 0.8}}}
            )

        expected_number_of_vectors = 7
        expected_number_of_documents = 5

        res = self.client.index(self.text_index_name).get_stats()
        assert res["numberOfDocuments"] == expected_number_of_documents
        assert res["numberOfVectors"] == expected_number_of_vectors

    def test_get_status_error(self):
        with self.assertRaises(MarqoWebError) as cm:
            self.client.index("A void index").get_stats()
        self. assertIn("index_not_found", str(cm.exception.message))