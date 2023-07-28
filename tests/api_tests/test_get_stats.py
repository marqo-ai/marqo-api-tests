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
        assert isinstance(res, dict)
        assert "numberOfVectors" in res
        assert "numberOfDocuments" in res

    def test_get_status_response_results(self):
        self.client.create_index(self.index_name)
        self.client.index(self.index_name).add_documents(
            documents=[
                {"description_1": "test-2", "description_2": "test"},  # 2 vectors
                {"description_1": "test-2", "description_2": "test", "description_3": "test"},  # 3 vectors
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

        res = self.client.index(self.index_name).get_stats()
        assert res["numberOfDocuments"] == expected_number_of_documents
        assert res["numberOfVectors"] == expected_number_of_vectors



