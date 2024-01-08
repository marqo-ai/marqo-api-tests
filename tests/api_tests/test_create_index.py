import uuid

from marqo.errors import MarqoWebError
from pytest import mark

from tests.marqo_test import MarqoTestCase


@mark.fixed
class TestCreateIndex(MarqoTestCase):

    def setUp(self) -> None:
        """As this test class is testing index creation,
        we need to create/delete index before/after each test"""
        super().setUp()
        self.index_name = "test_index"

    def tearDown(self):
        super().tearDown()
        try:
            self.client.delete_index(index_name=self.index_name)
        except MarqoWebError:
            pass

    def test_simple_index_creation(self):
        self.client.create_index(index_name=self.index_name)
        self.client.index(self.index_name).add_documents([{"test": "test"}], tensor_fields=["test"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        index_settings = self.client.index(self.index_name).get_settings()

        expected_settings = {
            'type': 'unstructured',
            'treatUrlsAndPointersAsImages': False,
            'filterStringMaxLength': 20,
            'model': 'hf/all_datasets_v4_MiniLM-L6',
            'normalizeEmbeddings': True,
            'textPreprocessing': {'split_length': 2, 'split_overlap': 0, 'split_method': 'sentence'},
            'imagePreprocessing': {},
            'vectorNumericType': 'float',
            'annParameters': {
                'spaceType': 'prenormalized-angular', 'parameters': {
                    'ef_construction': 128, 'm': 16}
            }
        }
        self.assertEqual(expected_settings, index_settings)

    def test_create_unstructured_image_index(self):
        self.client.create_index(index_name=self.index_name, type="unstructured",
                                 treat_urls_and_pointers_as_images=True, model="open_clip/ViT-B-32/laion400m_e32")
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]
        self.client.index(self.index_name).add_documents(documents, tensor_fields=["test", "image"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual(True, index_settings['treatUrlsAndPointersAsImages'])
        self.assertEqual("open_clip/ViT-B-32/laion400m_e32", index_settings['model'])

    def test_create_unstructured_text_index_custom_model(self):
        self.client.create_index(index_name=self.index_name, type="unstructured",
                                 treat_urls_and_pointers_as_images=False,
                                 model="test-model",
                                 model_properties={"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                                                   "dimensions": 384,
                                                   "tokens": 128,
                                                   "type": "sbert"}
                                 )
        documents = [{"test": "test"}]
        self.client.index(self.index_name).add_documents(documents, tensor_fields=["test"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual(False, index_settings['treatUrlsAndPointersAsImages'])
        self.assertEqual("test-model", index_settings['model'])
        self.assertEqual({"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                          "dimensions": 384,
                          "tokens": 128,
                          "type": "sbert"}, index_settings['modelProperties'])

    def test_created_unstructured_image_index_with_preprocessing(self):
        self.client.create_index(index_name=self.index_name, type="unstructured",
                                 treat_urls_and_pointers_as_images=True,
                                 model="open_clip/ViT-B-16/laion400m_e31",
                                 image_preprocessing={"patch_method": "simple"})
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]
        self.client.index(self.index_name).add_documents(documents, tensor_fields=["test", "image"])

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual(True, index_settings['treatUrlsAndPointersAsImages'])
        self.assertEqual("open_clip/ViT-B-16/laion400m_e31", index_settings['model'])
        self.assertEqual("simple", index_settings['imagePreprocessing']['patch_method'])

    def test_create_simple_structured_index(self):
        self.client.create_index(index_name=self.index_name, type="structured",
                                 model="hf/all_datasets_v4_MiniLM-L6",
                                 all_fields=[{"name": "test", "type": "text",
                                              "features": ["lexical_search"]}],
                                 tensor_fields=["test"])
        documents = [{"test": "test"}]
        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        expected_index_settings = {
            'type': 'structured',
            'allFields': [{'name': 'test', 'type': 'text', 'features': ['lexical_search']}],
            'tensorFields': ['test'],
            'model': 'hf/all_datasets_v4_MiniLM-L6',
            'normalizeEmbeddings': True,
            'textPreprocessing': {'split_length': 2, 'split_overlap': 0, 'split_method': 'sentence'},
            'imagePreprocessing': {},
            'vectorNumericType': 'float',
            'annParameters': {'spaceType': 'prenormalized-angular', 'parameters': {'ef_construction': 128, 'm': 16}}}
        self.assertEqual(expected_index_settings, index_settings)

    def test_create_structured_image_index(self):
        self.client.create_index(index_name=self.index_name,
                                 type="structured",
                                 model="open_clip/ViT-B-32/laion400m_e32",
                                 all_fields=[{"name": "test", "type": "text", "features": ["lexical_search"]},
                                             {"name": "image", "type": "image_pointer"}],
                                 tensor_fields=["test", "image"])
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]

        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()

        self.assertEqual(["test", "image"], index_settings["tensorFields"])
        self.assertEqual("open_clip/ViT-B-32/laion400m_e32", index_settings["model"])

    def test_create_structured_index_with_custom_model(self):
        self.client.create_index(index_name=self.index_name,
                                 type="structured",
                                 model="test-model",
                                 model_properties={"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                                                   "dimensions": 384,
                                                   "tokens": 128,
                                                   "type": "sbert"},
                                 all_fields=[{"name": "test", "type": "text", "features": ["lexical_search"]}],
                                 tensor_fields=["test"])
        documents = [{"test": "test"}]
        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))

        index_settings = self.client.index(self.index_name).get_settings()
        self.assertEqual("test-model", index_settings['model'])
        self.assertEqual({"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                          "dimensions": 384,
                          "tokens": 128,
                          "type": "sbert"}, index_settings['modelProperties'])

    def test_create_structured_image_index_with_preprocessing(self):
        self.client.create_index(index_name=self.index_name,
                                 type="structured",
                                 model="open_clip/ViT-B-16/laion400m_e31",
                                 image_preprocessing={"patch_method": "simple"},
                                 all_fields=[{"name": "test", "type": "text", "features": ["lexical_search"]},
                                             {"name": "image", "type": "image_pointer"}],
                                 tensor_fields=["test", "image"])
        image_url = "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"
        documents = [{"test": "test",
                      "image": image_url}]

        self.client.index(self.index_name).add_documents(documents)

        lexical_search_res = self.client.index(self.index_name).search(q="test", search_method="LEXICAL")
        tensor_search_res = self.client.index(self.index_name).search(q="test", search_method="TENSOR")
        tensor_search_res_image = self.client.index(self.index_name).search(q=image_url, search_method="TENSOR")

        self.assertEqual(1, len(lexical_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res['hits']))
        self.assertEqual(1, len(tensor_search_res_image['hits']))

        index_settings = self.client.index(self.index_name).get_settings()

        self.assertEqual(["test", "image"], index_settings["tensorFields"])
        self.assertEqual("open_clip/ViT-B-16/laion400m_e31", index_settings["model"])
        self.assertEqual("simple", index_settings['imagePreprocessing']['patch_method'])

    def test_dash_in_index_name_structured(self):
        index_name = "test-index-test-index" + str(uuid.uuid4())
        self.client.create_index(index_name, type="structured",
                                 all_fields=[{"name": "title", "type": "text"}],
                                 tensor_fields=["title"])
        self.indexes_to_delete.append(index_name)

    def test_dash_in_index_name_unstructured(self):
        index_name = "test-index-test-index" + str(uuid.uuid4())
        self.client.create_index(index_name, type="unstructured")
        self.indexes_to_delete.append(index_name)