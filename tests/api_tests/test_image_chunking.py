import uuid
from typing import Dict
import json

import numpy as np
import requests
from PIL import Image
from marqo.client import Client
from marqo.errors import MarqoApiError

from tests.marqo_test import MarqoTestCase


def generate_structured_index_settings_dict(index_name, image_preprocessing_method) -> Dict:
    return {
        "indexName": index_name,
        "type": "structured",
        "model": "open_clip/ViT-B-32/openai",
        "allFields": [{"name": "image_content", "type": "image_pointer"},
                      {"name": "text_content", "type": "text"}],
        "tensorFields": ["image_content", "text_content"],
        "imagePreprocessing": {"patchMethod": image_preprocessing_method}
    }


def generate_unstructured_index_settings_dict(index_name, image_preprocessing_method) -> Dict:
    return {
        "indexName": index_name,
        "type": "unstructured",
        "model": "open_clip/ViT-B-32/openai",
        "treatUrlsAndPointersAsImages": True,
        "imagePreprocessing": {"patchMethod": image_preprocessing_method}
    }


class TestUnstructuredImageChunking(MarqoTestCase):
    """Test for image chunking as a preprocessing step
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.unstructured_no_image_processing_index_name = (
                "unstructured_no_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.unstructured_simple_image_processing_index_name = (
                "unstructured_simple_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.unstructured_frcnn_image_processing_index_name = (
                "unstructured_frcnn_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))

        cls.structured_no_image_processing_index_name = (
                "structured_no_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_simple_image_processing_index_name = (
                "structured_simple_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))
        cls.structured_frcnn_image_processing_index_name = (
                "structured_frcnn_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))

        # create the structured indexes
        cls.create_indexes([
            generate_structured_index_settings_dict(cls.structured_no_image_processing_index_name, None),
            generate_structured_index_settings_dict(cls.structured_simple_image_processing_index_name, "simple"),
            generate_structured_index_settings_dict(cls.structured_frcnn_image_processing_index_name, "frcnn"),
        ])

        # create the unstructured indexes
        cls.create_indexes(
            [
                generate_unstructured_index_settings_dict(cls.unstructured_no_image_processing_index_name, None),
                generate_unstructured_index_settings_dict(cls.unstructured_simple_image_processing_index_name, "simple"),
                generate_unstructured_index_settings_dict(cls.unstructured_frcnn_image_processing_index_name, "frcnn"),
            ]
        )

        cls.indexes_to_delete = [
            cls.unstructured_no_image_processing_index_name,
            cls.unstructured_simple_image_processing_index_name,
            cls.unstructured_frcnn_image_processing_index_name,

            cls.structured_no_image_processing_index_name,
            cls.structured_simple_image_processing_index_name,
            cls.structured_frcnn_image_processing_index_name,
        ]

    def test_image_no_chunking(self):
        # image_size = (256, 384)
        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_no_image_processing_index_name, {"tensor_fields": ["image_content"]}, {},
             "unstructured_index, no image preprocessing"),
            (self.structured_no_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, no image preprocessing"),
        ]

        document = {
            '_id': '1', # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)
                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be the location
                self.assertEqual(temp_file_name, results['hits'][0]['_highlights']['image_content'])

    def test_image_simple_chunking(self):

        image_size = (256, 384)

        temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'

        test_case = [
            (self.unstructured_simple_image_processing_index_name, {"tensor_fields": ["image_content"]}, {},
             "unstructured_index, simple image preprocessing"),
            (self.structured_simple_image_processing_index_name, {}, {"searchable_attributes": ["image_content"]},
             "structured_index, simple image preprocessing"),
        ]

        document = {
            '_id': '1',  # '_id' can be provided but is not required
            'text_content': 'hello',
            'image_content': temp_file_name
        }

        for index_name, add_docs_call, search_call, msg in test_case:
            with self.subTest(msg):
                self.client.index(index_name).add_documents([document], **add_docs_call)

                # test the search works
                results = self.client.index(index_name).search('a', **search_call)

                self.assertEqual(temp_file_name, results['hits'][0]['image_content'])
                # the highlight should be a tuple with 4 elements representing
                r = results['hits'][0]['_highlights']['image_content']
                print(results['hits'][0]['_highlights']['image_content'])

    #
    # def test_image_simple_chunking_multifield(self):
    #
    #     image_size = (256, 384)
    #
    #     client = Client(**self.client_settings)
    #
    #     try:
    #         client.delete_index(self.index_name)
    #     except MarqoApiError as s:
    #         pass
    #
    #     settings = {
    #         "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
    #         "model":"ViT-B/16",
    #         "image_preprocessing_method":"simple"
    #         }
    #
    #     client.create_index(self.index_name, **settings)
    #
    #     temp_file_name_1 = 'https://avatars.githubusercontent.com/u/13092433?v=4' #brain
    #     temp_file_name_2 = 'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png' #hippo
    #
    #     img = Image.open(requests.get(temp_file_name_1, stream=True).raw)
    #
    #
    #     documents = [{'_id': '1', # '_id' can be provided but is not required
    #         'attributes': 'hello',
    #         'description': 'the image chunking can (optionally) chunk the image into sub-patches (aking to segmenting text) by using either a learned model or simple box generation and cropping',
    #         'location': temp_file_name_1},
    #         {'_id': '11', # '_id' can be provided but is not required
    #         'attributes': 'hello sdsd ' ,
    #         'description': 'the im sds age csdsdssdsddhunking can (optionally) chunk the image into sub-patches (aking to segmenting text) by using either a learned model or simple box generation and cropping',
    #         'location': temp_file_name_1,
    #         'location_1': temp_file_name_2},
    #         {'_id': '2', # '_id' can be provided but is not required
    #         'attributes': 'hello',
    #         'description': 'the imo segmenting text) by using either a learned model or simple box generation and cropping'},
    #         {'_id': '3', # '_id' can be provided but is not required
    #         'description': 'sds s ds',
    #         'location_1': temp_file_name_2},
    #     ]
    #
    #     client.index(self.index_name).add_documents(documents, non_tensor_fields=[], auto_refresh=True)
    #
    #     # test the search works
    #     results = client.index(self.index_name).search('brain', searchable_attributes=['location', 'location_1'])
    #     print(results)
    #     assert results['hits'][0]['location'] == temp_file_name_1, results
    #
    #     # search only the image location
    #     results = client.index(self.index_name).search('hippo', searchable_attributes=['location', 'location_1'])
    #     print(results)
    #     assert results['hits'][0]['location_1'] == temp_file_name_2
    #     # the highlight should be the location
    #     assert results['hits'][0]['_highlights']['location_1'] != temp_file_name_2
    #     assert len(results['hits'][0]['_highlights']['location_1']) == 4
    #     assert all(isinstance(_n, (float, int)) for _n in results['hits'][0]['_highlights']['location_1'])
    #
    #     # search using the image itself, should return a full sized image as highlight
    #     results = client.index(self.index_name).search(temp_file_name_1)
    #     print(results)
    #     assert abs(np.array(results['hits'][0]['_highlights']['location']) - np.array([0, 0, img.size[0], img.size[1]])).sum() < 1e-6
    #
    # def test_image_yolox_chunking(self):
    #
    #     image_size = (256, 384)
    #
    #     client = Client(**self.client_settings)
    #
    #     try:
    #         client.delete_index(self.index_name)
    #     except MarqoApiError as s:
    #         pass
    #
    #     settings = {
    #         "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
    #         "model":"ViT-B/16",
    #         "image_preprocessing_method":"marqo-yolo"
    #         }
    #
    #     client.create_index(self.index_name, **settings)
    #
    #
    #     temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'
    #
    #     img = Image.open(requests.get(temp_file_name, stream=True).raw)
    #
    #     document1 = {'_id': '1', # '_id' can be provided but is not required
    #         'attributes': 'hello',
    #         'description': 'the image chunking can (optionally) chunk the image into sub-patches (akin to segmenting text) by using either a learned model or simple box generation and cropping',
    #         'location': temp_file_name}
    #
    #     client.index(self.index_name).add_documents([document1], non_tensor_fields=[], auto_refresh=True)
    #
    #     # test the search works
    #     results = client.index(self.index_name).search('a')
    #     print(results)
    #     assert results['hits'][0]['location'] == temp_file_name
    #
    #     # search only the image location
    #     results = client.index(self.index_name).search('a', searchable_attributes=['location'])
    #     print(results)
    #     assert results['hits'][0]['location'] == temp_file_name
    #     # the highlight should be the location
    #     assert results['hits'][0]['_highlights']['location'] != temp_file_name
    #     assert len(results['hits'][0]['_highlights']['location']) == 4
    #     assert all(isinstance(_n, (float, int)) for _n in results['hits'][0]['_highlights']['location'])
    #
    #     # search using the image itself, should return a full sized image as highlight
    #     results = client.index(self.index_name).search(temp_file_name)
    #     print(results)
    #     assert abs(np.array(results['hits'][0]['_highlights']['location']) - np.array([0, 0, img.size[0], img.size[1]])).sum() < 1e-6
    #
    # def test_image_dino_chunking(self):
    #
    #     image_size = (256, 384)
    #
    #     client = Client(**self.client_settings)
    #
    #     try:
    #         client.delete_index(self.index_name)
    #     except MarqoApiError as s:
    #         pass
    #
    #     settings = {
    #         "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
    #         "model":"ViT-B/16",
    #         "image_preprocessing_method":"dino-v1"
    #         }
    #
    #     client.create_index(self.index_name, **settings)
    #
    #
    #     temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'
    #
    #     img = Image.open(requests.get(temp_file_name, stream=True).raw)
    #
    #     document1 = {'_id': '1', # '_id' can be provided but is not required
    #         'attributes': 'hello',
    #         'description': 'the image chunking can (optionally) chunk the image into sub-patches (akin to segmenting text) by using either a learned model or simple box generation and cropping',
    #         'location': temp_file_name}
    #
    #     client.index(self.index_name).add_documents([document1], non_tensor_fields=[], auto_refresh=True)
    #
    #     # test the search works
    #     results = client.index(self.index_name).search('a')
    #     print(results)
    #     assert results['hits'][0]['location'] == temp_file_name
    #
    #     # search only the image location
    #     results = client.index(self.index_name).search('a', searchable_attributes=['location'])
    #     print(results)
    #     assert results['hits'][0]['location'] == temp_file_name
    #     # the highlight should be the location
    #     assert results['hits'][0]['_highlights']['location'] != temp_file_name
    #     assert len(results['hits'][0]['_highlights']['location']) == 4
    #     assert all(isinstance(_n, (float, int)) for _n in results['hits'][0]['_highlights']['location'])
    #
    #     # search using the image itself, should return a full sized image as highlight
    #     results = client.index(self.index_name).search(temp_file_name)
    #     print(results)
    #     assert abs(np.array(results['hits'][0]['_highlights']['location']) - np.array([0, 0, img.size[0], img.size[1]])).sum() < 1e-6
    #
    #     try:
    #         client.delete_index(self.index_name)
    #     except MarqoApiError as s:
    #         pass
    #
    #     settings = {
    #         "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
    #         "model":"ViT-B/16",
    #         "image_preprocessing_method":"dino-v2"
    #         }
    #
    #     client.create_index(self.index_name, **settings)
    #
    #
    #     temp_file_name = 'https://avatars.githubusercontent.com/u/13092433?v=4'
    #
    #     img = Image.open(requests.get(temp_file_name, stream=True).raw)
    #
    #     document1 = {'_id': '1', # '_id' can be provided but is not required
    #         'attributes': 'hello',
    #         'description': 'the image chunking can (optionally) chunk the image into sub-patches (akin to segmenting text) by using either a learned model or simple box generation and cropping',
    #         'location': temp_file_name}
    #
    #     client.index(self.index_name).add_documents([document1], non_tensor_fields=[], auto_refresh=True)
    #
    #     # test the search works
    #     results = client.index(self.index_name).search('a')
    #     print(results)
    #     assert results['hits'][0]['location'] == temp_file_name
    #
    #     # search only the image location
    #     results = client.index(self.index_name).search('a', searchable_attributes=['location'])
    #     print(results)
    #     assert results['hits'][0]['location'] == temp_file_name
    #     # the highlight should be the location
    #     assert results['hits'][0]['_highlights']['location'] != temp_file_name
    #     assert len(results['hits'][0]['_highlights']['location']) == 4
    #     assert all(isinstance(_n, (float, int)) for _n in results['hits'][0]['_highlights']['location'])
    #
    #     # search using the image itself, should return a full sized image as highlight
    #     results = client.index(self.index_name).search(temp_file_name)
    #     print(results)
    #     assert abs(np.array(results['hits'][0]['_highlights']['location']) - np.array([0, 0, img.size[0], img.size[1]])).sum() < 1e-6
    #
        