import copy
import pprint
from urllib.parse import quote_plus
from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError, MarqoWebError
import unittest
from tests.marqo_test import MarqoTestCase
from marqo import enums
from unittest import mock
import numpy as np
import pytest
import requests
import json


class TestAddDocuments(MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    # Create index tests
    def test_create_index(self):
        self.client.create_index(index_name=self.index_name_1)

    def test_create_index_double(self):
        self.client.create_index(index_name=self.index_name_1)
        try:
            self.client.create_index(index_name=self.index_name_1)
        except MarqoWebError as e:
            assert "index_already_exists" == e.code

    def test_create_index_hnsw(self):
        self.client.create_index(index_name=self.index_name_1, settings_dict={
            "index_defaults": {
                "ann_parameters": {
                    "parameters": {
                        "m": 24
                    }
                }
            }
        })
        assert self.client.get_index(self.index_name_1).get_settings() \
                   ["index_defaults"]["ann_parameters"]["parameters"]["m"] == 24

        # Ensure non-specified values are in default
        assert self.client.get_index(self.index_name_1).get_settings() \
                   ["index_defaults"]["ann_parameters"]["parameters"]["ef_construction"] == 128
        assert self.client.get_index(self.index_name_1).get_settings() \
                   ["index_defaults"]["ann_parameters"]["space_type"] == "cosinesimil"

    # Delete index tests:

    def test_delete_index(self):
        self.client.create_index(index_name=self.index_name_1)
        self.client.delete_index(self.index_name_1)
        self.client.create_index(index_name=self.index_name_1)

    # Get index tests:

    def test_get_index(self):
        self.client.create_index(index_name=self.index_name_1)
        index = self.client.get_index(self.index_name_1)
        assert index.index_name == self.index_name_1

    def test_get_index_non_existent(self):
        try:
            index = self.client.get_index("some-non-existent-index")
            raise AssertionError
        except MarqoWebError as e:
            assert e.code == "index_not_found"

    # Add documents tests:

    def test_add_documents_with_ids(self):
        self.client.create_index(index_name=self.index_name_1)
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
        res = self.client.index(self.index_name_1).add_documents([
            d1, d2
        ], non_tensor_fields=[])
        retrieved_d1 = self.client.index(self.index_name_1).get_document(document_id="e197e580-0393-4f4e-90e9-8cdf4b17e339")
        assert retrieved_d1 == d1
        retrieved_d2 = self.client.index(self.index_name_1).get_document(document_id="123456")
        assert retrieved_d2 == d2

    def test_add_documents(self):
        """indexes the documents and retrieves the documents with the generated IDs"""
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
            "doc title": "Cool Document 1",
            "field 1": "some extra info"
        }
        d2 = {
                "doc title": "Just Your Average Doc",
                "field X": "this is a solid doc"
            }
        res = self.client.index(self.index_name_1).add_documents([d1, d2], non_tensor_fields=[])
        ids = [item["_id"] for item in res["items"]]
        assert len(ids) == 2
        assert ids[0] != ids[1]
        retrieved_d0 = self.client.index(self.index_name_1).get_document(ids[0])
        retrieved_d1 = self.client.index(self.index_name_1).get_document(ids[1])
        del retrieved_d0["_id"]
        del retrieved_d1["_id"]
        assert retrieved_d0 == d1 or retrieved_d0 == d2
        assert retrieved_d1 == d1 or retrieved_d1 == d2

    def test_add_batched_documents(self):
        self.client.create_index(self.index_name_1)
        ix = self.client.index(index_name=self.index_name_1)
        doc_ids = [str(num) for num in range(0, 100)]
        docs = [
            {"Title": f"The Title of doc {doc_id}",
             "Generic text": "some text goes here...",
             "_id": doc_id}
            for doc_id in doc_ids]

        ix.add_documents(docs, client_batch_size=10, tensor_fields=["Title", "Generic text"])
        ix.refresh()
        # TODO we should do a count in here...
        # takes too long to search for all
        for _id in [0, 19, 20, 99]:
            original_doc = docs[_id].copy()
            assert ix.get_document(document_id=str(_id)) == original_doc

    def test_add_documents_with_ids_twice(self):
        self.client.create_index(index_name=self.index_name_1)
        d1 = {
            "doc title": "Just Your Average Doc",
            "field X": "this is a solid doc",
            "_id": "56"
        }
        self.client.index(self.index_name_1).add_documents([d1], non_tensor_fields=[])
        assert d1 == self.client.index(self.index_name_1).get_document("56")
        d2 = {
            "_id": "56",
            "completely": "different doc.",
            "field X": "this is a solid doc"
        }
        self.client.index(self.index_name_1).add_documents([d2], non_tensor_fields=[])
        assert d2 == self.client.index(self.index_name_1).get_document("56")

    def test_add_documents_long_fields(self):
        """TODO
        """

    def test_update_docs_updates_chunks(self):
        """TODO"""
    # get documents tests :

    def test_get_document(self):
        """FIXME (do edge cases)"""

    # user experience tests:

    def test_add_documents_missing_index_fails(self):
        with pytest.raises(MarqoWebError) as ex:
            self.client.index(self.index_name_1).add_documents([{"abd": "efg"}], non_tensor_fields=[])

        assert "index_not_found" == ex.value.code

    def test_add_documents_with_device(self):
        temp_client = copy.deepcopy(self.client)

        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            temp_client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"}, {"d2", "some data"}
            ], device="cuda:45", non_tensor_fields=[])
            return True

        assert run()

        args, kwargs = mock__post.call_args
        assert "device=cuda45" in kwargs["path"]

    def test_add_documents_no_device(self):
        """No device should be in path if no device is set
        """
        temp_client = copy.deepcopy(self.client)

        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            temp_client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"}, {"d2", "some data"}
            ], non_tensor_fields=[])
            return True

        assert run()

        args, kwargs = mock__post.call_args
        assert "device" not in kwargs["path"]

    def test_add_documents_set_refresh(self):
        temp_client = copy.deepcopy(self.client)

        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            temp_client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"}, {"d2", "some data"}
            ], auto_refresh=False, non_tensor_fields=[])
            temp_client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"}, {"d2", "some data"}
            ], auto_refresh=True, non_tensor_fields=[])
            return True

        assert run()

        args, kwargs0 = mock__post.call_args_list[0]
        assert "refresh=false" in kwargs0["path"]
        args, kwargs1 = mock__post.call_args_list[1]
        assert "refresh=true" in kwargs1["path"]

    def test_add_documents_with_no_processes(self):
        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            self.client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"}, {"d2", "some data"}
            ], non_tensor_fields=[])
            return True

        assert run()

        args, kwargs = mock__post.call_args
        assert "processes=12" not in kwargs["path"]

    def test_add_documents_empty(self):
        """
        Test that adding an empty list of documents fails with bad_request
        """
        self.client.create_index(index_name=self.index_name_1)
        try:
            self.client.index(self.index_name_1).add_documents(documents=[], non_tensor_fields=[])
            raise AssertionError
        except MarqoWebError as e:
            assert "bad_request" == e.code

    def test_add_documents_deprecated_query_parameters_and_new_api(self):
        """This test is to ensure that the new API does not accept old query parameters"""
        self.client.create_index(self.index_name_1)
        model_auth = {
            's3': {
                "aws_access_key_id": "<SOME ACCESS KEY ID>",
                "aws_secret_access_key": "<SOME SECRET ACCESS KEY>"
            }
        }

        mappings = {
            "combo_text_image":
                {"type": "multimodal_combination",
                 "weights":
                     {
                         "my_text_attribute_1": 0.5,
                         "my_image_attribute_1": 0.5,
                     }
                 }
        }

        image_download_headers = {
            "my-image-store-api-key": "some-super-secret-image-store-key"
        },

        deprecated_query_parameters_list = ["non_tensor_fields=Title&non_tensor_fields=Genre",
                                            "use_existing_tensors=true"]
                                            # f"model_auth={quote_plus(json.dumps(model_auth))}",
                                            # f"mappings={quote_plus(json.dumps(mappings))}",
                                            # f"image_download_headers={quote_plus(json.dumps(image_download_headers))}"]

        data = {
            "documents": [
                {
                    "Title": "The Travels of Marco Polo",
                    "Description": "A 13th-century travelogue describing the travels of Polo",
                    "Genre": "History"
                },
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection",
                    "_id": "article_591",
                    "Genre": "Science"
                }
            ],
            "nonTensorFields": ["Title", "Genre"]
        }

        for deprecated_query_parameters in deprecated_query_parameters_list:
            url = f"{self.authorized_url}/indexes/{self.index_name_1}/documents?{deprecated_query_parameters}"
            headers = {'Content-type': 'application/json'}

            response = requests.post(url, headers=headers, data=json.dumps(data))
            assert str(response.status_code).startswith("4")
            self.assertIn("Marqo is not accepting any of the following parameters in the query string",
                          str(response.json()))

    def test_add_documents_extra_field(self):
        """This test is to ensure that the new API does not accept extra body parameters"""
        self.client.create_index(self.index_name_1)
        url = f'{self.authorized_url}/{self.index_name_1}/documents'
        headers = {'Content-type': 'application/json'}

        data = {
            "documents": [
                {
                    "Title": "The Travels of Marco Polo",
                    "Description": "A 13th-century travelogue describing the travels of Polo",
                    "Genre": "History"
                },
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection",
                    "_id": "article_591",
                    "Genre": "Science"
                }
            ],
            "non_TensorFields": ["Title", "Genre"]  # not permitted field
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        assert str(response.status_code).startswith("4")
        self.assertIn("extra fields not permitted", str(response.json()))

    def test_add_docs_image_download_headers(self):
        mock__post = mock.MagicMock()

        @mock.patch("marqo._httprequests.HttpRequests.post", mock__post)
        def run():
            image_download_headers = {"Authentication": "my-secret-key"}
            self.client.index(index_name=self.index_name_1).add_documents(
                documents=[{"some": "data"}], image_download_headers=image_download_headers,
            non_tensor_fields=[])
            args, kwargs = mock__post.call_args
            assert "imageDownloadHeaders" in kwargs['body']
            assert kwargs['body']['imageDownloadHeaders'] == image_download_headers

            return True

        assert run()


@pytest.mark.cpu_only_test
class TestAddDocumentsCPUOnly(MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def tearDown(self) -> None:
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def test_add_documents_defaults_to_cpu(self):
        """
            Ensures that when cuda is NOT available, when we send an add docs request with no device,
            cuda is selected as default and used for this.
        """
        index_settings = {
            "index_defaults": {
                # model was chosen due to bigger difference between cuda and cpu vectors
                "model": "open_clip/ViT-B-32-quickgelu/laion400m_e31",
                "normalize_embeddings": True
            }
        }

        self.client.create_index(self.index_name_1, settings_dict=index_settings)

        self.client.index(self.index_name_1).add_documents([{"_id": "explicit_cpu", "title": "blah"}], device="cpu",
                                                           non_tensor_fields=[])
        self.client.index(self.index_name_1).add_documents([{"_id": "default_device", "title": "blah"}],
                                                           non_tensor_fields=[])
        
        cpu_vec = self.client.index(self.index_name_1).get_document(document_id="explicit_cpu", expose_facets=True)['_tensor_facets'][0]["_embedding"]
        default_vec = self.client.index(self.index_name_1).get_document(document_id="default_device", expose_facets=True)['_tensor_facets'][0]["_embedding"]

        # Confirm that CPU was used by default.
        # CPU-computed vectors are slightly different from CUDA-computed vectors
        assert np.allclose(np.array(cpu_vec), np.array(default_vec), atol=1e-5)

    def test_add_documents_device_not_available(self):
        """
            Ensures that when cuda is NOT available, an error is thrown when trying to use cuda
        """
        self.client.create_index(self.index_name_1)

        # Add docs with CUDA must fail if CUDA is not available
        try:
            self.client.index(self.index_name_1).add_documents([{"_id": "explicit_cuda", "title": "blah"}],
                                                               device="cuda", non_tensor_fields=[])
            raise AssertionError
        except MarqoWebError:
            pass
