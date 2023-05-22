import logging
import subprocess
import os
import time
from requests import HTTPError
from tests import marqo_test
from tests import utilities
from marqo import Client
from marqo.errors import MarqoApiError, BackendCommunicationError, MarqoWebError
import pprint
import json

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
            print("Marqo Web Error correctly raised")
            pass
        
        # Rerun marqo with new replica count
        max_replicas = 5
        print(f"Attempting to rerun marqo with max replicas: {max_replicas}")
        utilities.rerun_marqo_with_env_vars(
            env_vars = f"-e MARQO_MAX_NUMBER_OF_REPLICAS={max_replicas}"
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
    

    def test_preload_models(self):
        # check preloaded models (should be default)
        default_models = ["'hf/all_datasets_v4_MiniLM-L6", "ViT-L/14"]
        res = self.client.get_loaded_models()
        assert set([item["model_name"] for item in res["models"]]) == set(default_models)

        # Rerun marqo with new custom model
        open_clip_model_object = {
            "model": "open-clip-1",
            "model_properties": {
                "name": "ViT-B-32-quickgelu",
                "dimensions": 512,
                "type": "open_clip",
                "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
            }
        }

        print(f"Attempting to rerun marqo with custom model {open_clip_model_object['model']}")
        utilities.rerun_marqo_with_env_vars(
            env_vars = f"-e MARQO_MODELS_TO_PRELOAD=[{json.dumps(open_clip_model_object)}]"
        )

        # check preloaded models (should be custom model)
        custom_models = ["open-clip-1"]
        res = self.client.get_loaded_models()
        assert set([item["model_name"] for item in res["models"]]) == set(custom_models)
