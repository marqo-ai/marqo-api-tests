from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError, MarqoWebError
from tests.marqo_test import MarqoTestCase
from parameterized import parameterized


class TestGenericModelSupport(MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    @parameterized.expand([
        ['model_not_in_registry', {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                                                         "dimensions": 384,
                                                         "tokens": 128,
                                                         "type": "sbert"}],
        ['model_in_registry', {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                                                     "dimensions": 384,
                                                     "tokens": 128,
                                                     "type": "sbert",
                                                     "notes": ""}]
    ])
    def test_vector_search_with_generic_models(self, name, model_properties):

        settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "try-model",
                "model_properties": model_properties,
                "normalize_embeddings": True,
            }
        }
        self.client.create_index(index_name=self.index_name_1, settings_dict=settings)

        d1 = {
            "doc title": "Cool Document 1",
            "field 1": "some extra info",
            "_id": "e197e580-0393-4f4e-90e9-8cdf4b17e339"
        }
        d2 = {
            "doc title": "Just Your Average Doc",
            "field X": "this is a solid doc",
            "_id": "123456"
        }

        self.client.index(self.index_name_1).add_documents([d1, d2])

        search_res = self.client.index(self.index_name_1).search("cool document")

        assert len(search_res["hits"]) == 1
        assert self.strip_marqo_fields(search_res["hits"][0]) == d1
