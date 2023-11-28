"""Please have a running Marqo instance to test against!

Pass its settings to local_marqo_settings.
"""
from typing import List, Dict
import json

import unittest
from marqo.utils import construct_authorized_url
from marqo import Client
from marqo.errors import MarqoWebError
import requests


class MarqoTestCase(unittest.TestCase):

    indexes_to_delete = []
    _MARQO_URL = "http://localhost:8882"

    @classmethod
    def setUpClass(cls) -> None:
        local_marqo_settings = {
            "url": cls._MARQO_URL
        }
        cls.client_settings = local_marqo_settings
        cls.authorized_url = cls.client_settings["url"]

    @classmethod
    def tearDownClass(cls) -> None:
        # A function that will be automatically called after each test call
        # This removes all the loaded models to save memory space.
        cls.removeAllModels()

    @classmethod
    def create_indexes(cls, index_names_and_settings: List[Dict]):
        """A function to call the internal Marqo API to create a batch of indexes.

        Args:
            index_names_and_settings: list of dictionaries, each dictionary
                containing the name of the index and its settings
            [{"index_name": "index1", "settings_dict": {"type": "unstructured"}}, ...]
        """

        index_names: List[str] = [index_name_and_settings["index_name"] for index_name_and_settings in index_names_and_settings]
        settings_dict_list: List[Dict] = [index_name_and_settings["settings_dict"] for index_name_and_settings in index_names_and_settings]

        body = {
            "index_names": index_names,
            "index_settings_list": settings_dict_list,
        }

        r = requests.post(f"{cls._MARQO_URL}/internal/indexes/create-batch", json=body)

    @classmethod
    def delete_indexes(cls, index_names: List[str]):
        r = requests.post(f"{cls._MARQO_URL}/internal/indexes/delete-batch", data=json.dumps(index_names))

    @classmethod
    def clear_indexes(cls, index_names: List[str]):
        r = requests.post(f"{cls._MARQO_URL}/internal/indexes/documents/delete-batch", data=json.dumps(index_names))


    @classmethod
    def tearDownClass(cls) -> None:
        if cls.indexes_to_delete:
            cls.delete_indexes(cls.indexes_to_delete)


    @classmethod
    def removeAllModels(cls) -> None:
        # A function that can be called to remove loaded models in Marqo.
        # Use it whenever you think there is a risk of OOM problem.
        # E.g., add it into the `tearDown` function to remove models between test cases.
        client = Client(**cls.client_settings)
        index_names = [index.index_name for index in client.get_indexes()["results"]]
        for index_name in index_names:
            loaded_models = client.index(index_name).get_loaded_models().get("models", [])
            for model in loaded_models:
                try:
                    client.index(index_name).eject_model(model_name=model["model_name"], model_device=model["model_device"])
                except MarqoWebError:
                    pass

