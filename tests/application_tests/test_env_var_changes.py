import logging
import subprocess
import time
from requests import HTTPError
from tests import marqo_test
from marqo import Client
from marqo.errors import MarqoApiError, BackendCommunicationError, MarqoWebError


class TestEnvVarChanges(marqo_test.MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def test_max_replicas(self):

        # Attempt to create index with 4 replicas (should fail, since default max is 1)
        try:
            res_0 = self.client.create_index(index_name=self.index_name_1, settings_dict={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": True,
                    "model": "ViT-B/32",
                },
                "number_of_replicas": 4
            })
            raise AssertionError()
        except MarqoWebError as e:
            pass
        
        # Rerun marqo with new replica count
        max_replicas = 5
        print(f"Attempting to rerun marqo with max replicas: {max_replicas}")
        rerun_marqo_with_env_vars(
            env_vars = f"-e MARQO_MAX_CONCURRENT_INDEX='{max_replicas}'"
        )

        # Attempt to create index with 4 replicas (should succeed)
        res_0 = self.client.create_index(index_name=self.index_name_1, settings_dict={
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "ViT-B/32",
            },
            "number_of_replicas": 4
        })

        # Make sure new index has 4 replicas
        assert self.client.get_index(self.index_name_1).get_settings() \
            ["number_of_replicas"] == 4

