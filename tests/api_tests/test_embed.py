import uuid

import numpy as np

from tests.marqo_test import MarqoTestCase


class TestEmbed(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_index_name = "structured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_non_e5 = "unstructured_non_e5_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_languagebind_index_name = "unstructured_languagebind_" + str(uuid.uuid4()).replace('-', '')
        cls.structured_languagebind_index_name = "structured_languagebind_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "text_field_1", "type": "text"},
                    {"name": "text_field_2", "type": "text"}
                ],
                "tensorFields": ["text_field_1", "text_field_2"]
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
            },
            {
                "indexName": cls.unstructured_index_non_e5,
                "type": "unstructured",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            {
                "indexName": cls.unstructured_languagebind_index_name,
                "type": "unstructured",
                "model": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",
                "treatUrlsAndPointersAsMedia": True,
                "treatUrlsAndPointersAsImages": True
            },
            {
                "indexName": cls.structured_languagebind_index_name,
                "type": "structured",
                "model": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",
                "allFields": [
                    {"name": "text_field", "type": "text"},
                    {"name": "video_field", "type": "video_pointer"},
                    {"name": "audio_field", "type": "audio_pointer"},
                    {"name": "image_field", "type": "image_pointer"}
                ],
                "tensorFields": ["text_field", "video_field", "audio_field", "image_field"]
            }
        ])
        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name, cls.unstructured_index_non_e5,
                                 cls.unstructured_languagebind_index_name, cls.structured_languagebind_index_name]

    def test_embed_single_string(self):
        """Embeds a string. Use add docs and get docs with tensor facets to ensure the vector is correct.
                Checks the basic functionality and response structure"""

        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_d1 = self.client.index(test_index_name).get_document(
                    document_id="doc1", expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed("Jimmy Butler is the GOAT.", device="cpu")
                else:
                    embed_res = self.client.index(test_index_name).embed("Jimmy Butler is the GOAT.", device="cpu", content_type="document")

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], "Jimmy Butler is the GOAT.")
                self.assertTrue(np.allclose(embed_res["embeddings"][0], retrieved_d1["_tensor_facets"][0] ["_embedding"], atol=1e-6))


    def test_embed_with_device(self):
        """Embeds a string with device parameter. Use add docs and get docs with tensor facets to ensure the vector is correct.
                        Checks the basic functionality and response structure"""
        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_d1 = self.client.index(test_index_name).get_document(
                    document_id="doc1", expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed(content="Jimmy Butler is the GOAT.", device="cpu")
                else:
                    embed_res = self.client.index(test_index_name).embed(content="Jimmy Butler is the GOAT.", device="cpu", content_type="document")

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], "Jimmy Butler is the GOAT.")
                self.assertTrue(np.allclose(embed_res["embeddings"][0], retrieved_d1["_tensor_facets"][0] ["_embedding"], atol=1e-6))

    def test_embed_single_dict(self):
        """Embeds a dict. Use add docs and get docs with tensor facets to ensure the vector is correct.
                        Checks the basic functionality and response structure"""
        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_d1 = self.client.index(test_index_name).get_document(
                    document_id="doc1", expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed(content={"Jimmy Butler is the GOAT.": 1})
                else:
                    embed_res = self.client.index(test_index_name).embed(content={"Jimmy Butler is the GOAT.": 1}, content_type="document")

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], {"Jimmy Butler is the GOAT.": 1})
                self.assertTrue(np.allclose(embed_res["embeddings"][0], retrieved_d1["_tensor_facets"][0] ["_embedding"], atol=1e-6))

    def test_embed_list_content(self):
        """Embeds a list with string and dict. Use add docs and get docs with tensor facets to ensure the vector is correct.
                                Checks the basic functionality and response structure"""
        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                d2 = {
                    "_id": "doc2",
                    "text_field_1": "Alex Caruso is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1, d2], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_docs = self.client.index(test_index_name).get_documents(
                    document_ids=["doc1", "doc2"], expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed(
                        content=[{"Jimmy Butler is the GOAT.": 1}, "Alex Caruso is the GOAT."],
                    )
                else:
                    embed_res = self.client.index(test_index_name).embed(
                        content=[{"Jimmy Butler is the GOAT.": 1}, "Alex Caruso is the GOAT."],
                        content_type="document"
                    )

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], [{"Jimmy Butler is the GOAT.": 1}, "Alex Caruso is the GOAT."])
                self.assertTrue(
                    np.allclose(embed_res["embeddings"][0], retrieved_docs["results"][0]["_tensor_facets"][0]["_embedding"], atol=1e-6))
                self.assertTrue(
                    np.allclose(embed_res["embeddings"][1], retrieved_docs["results"][1]["_tensor_facets"][0]["_embedding"], atol=1e-6))
                
    def test_embed_languagebind_images(self):
        """Test embedding images using LanguageBind model"""
        image_urls = [
            "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
        ]

        embed_res = self.client.index(self.unstructured_languagebind_index_name).embed(content=image_urls)

        self.assertIn("processingTimeMs", embed_res)
        self.assertEqual(embed_res["content"], image_urls)
        self.assertEqual(len(embed_res["embeddings"]), 3)
        
        # Check if embeddings are non-zero and have the expected shape
        for embedding in embed_res["embeddings"]:
            self.assertTrue(len(embedding) > 0)
            self.assertTrue(np.any(embedding))

        # Check that embeddings are close to the expected values
        expected_embedding = [0.019889963790774345, -0.01263524405658245,
                              0.026028314605355263, 0.005291664972901344, -0.013181567192077637]
        for embedding in embed_res["embeddings"]:
            for i, value in enumerate(expected_embedding):
                self.assertAlmostEqual(embedding[i], value, places=5)

    def test_embed_languagebind_videos(self):
        """Test embedding videos using LanguageBind model"""
        video_urls = [
            "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4"
        ]

        embed_res = self.client.index(self.unstructured_languagebind_index_name).embed(content=video_urls)

        self.assertIn("processingTimeMs", embed_res)
        self.assertEqual(embed_res["content"], video_urls)
        self.assertEqual(len(embed_res["embeddings"]), 3)
        
        # Check if embeddings are non-zero and have the expected shape
        for embedding in embed_res["embeddings"]:
            self.assertTrue(len(embedding) > 0)
            self.assertTrue(np.any(embedding))

        # Check that embeddings are close to the expected values
        expected_embedding = [0.0394694060087204, 0.049264926463365555,
                              -0.014714145101606846, 0.05715121701359749, -0.019508328288793564]
        for embedding in embed_res["embeddings"]:
            for i, value in enumerate(expected_embedding):
                self.assertAlmostEqual(embedding[i], value, places=5)