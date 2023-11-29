import uuid

from tests.marqo_test import MarqoTestCase
from marqo.client import Client


class TestGetSettings(MarqoTestCase):
    default_index_name = "default_index" + str(uuid.uuid4()).replace('-', '')
    custom_index_name = "custom_index" + str(uuid.uuid4()).replace('-', '')

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.delete_indexes([cls.default_index_name, cls.custom_index_name])

    def test_default_settings(self):
        """default fields should be returned if index is created with default settings
            sample structure of output: {'index_defaults': {'treat_urls_and_pointers_as_images': False,
                                          'text_preprocessing': {'split_method': 'sentence', 'split_length': 2,
                                                                 'split_overlap': 0},
                                          'model': 'hf/all_datasets_v4_MiniLM-L6', 'normalize_embeddings': True,
                                          'image_preprocessing': {'patch_method': None}}, 'number_of_shards': 5,
                                          'number_of_replicas' : 1,}
        """
        self.client.create_index(index_name=self.default_index_name, type="structured",
                                 all_fields=[{"name": "title", "type": "text"}],tensor_fields=["title"])

        ix = self.client.index(self.default_index_name)
        index_settings = ix.get_settings()
        fields = {'text_preprocessing', 'model', 'normalize_embeddings',
                  'image_preprocessing', "all_fields", "tensor_fields"}
        self.assertTrue(fields.issubset(set(index_settings)))

    def test_custom_settings(self):
        """adding custom settings to the index should be reflected in the returned output
        """
        model_properties = {'name': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                            'dimensions': 384,
                            'tokens': 128,
                            'type': 'sbert'}

        index_settings = {
            "type": "structured",
            'model': 'test-model',
            'model_properties': model_properties,
            'normalize_embeddings': True,
            "all_fields": [
                {"name": "title", "type": "text"},
                {"name": "content", "type": "text"},
            ],
            "tensor_fields": ["title", "content"],
        }

        res = self.client.create_index(index_name=self.custom_index_name, settings_dict=index_settings)

        ix = self.client.index(self.custom_index_name)
        index_settings = ix.get_settings()
        fields = {'text_preprocessing', 'model', 'normalize_embeddings',
                  'image_preprocessing', 'model_properties', "all_fields", "tensor_fields"}
        self.assertTrue(fields.issubset(set(index_settings)))