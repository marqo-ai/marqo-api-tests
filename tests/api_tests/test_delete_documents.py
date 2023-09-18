import copy
import pprint
from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError, MarqoWebError
import unittest
from tests.marqo_test import MarqoTestCase
from marqo import enums
from unittest import mock
import numpy as np
import pytest

class TestDeleteDocuments(MarqoTestCase):

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

    def test_delete_docs(self):
        self.client.create_index(index_name=self.index_name_1)
        self.client.index(self.index_name_1).add_documents([
            {"abc": "wow camel", "_id": "123"},
            {"abc": "camels are cool", "_id": "foo"}
        ], non_tensor_fields=[], auto_refresh=True)
        res0 = self.client.index(self.index_name_1).search("wow camel")
        assert res0['hits'][0]["_id"] == "123"
        assert len(res0['hits']) == 2
        self.client.index(self.index_name_1).delete_documents(["123"], auto_refresh=True)
        res1 = self.client.index(self.index_name_1).search("wow camel")
        assert res1['hits'][0]["_id"] == "foo"
        assert len(res1['hits']) == 1

    def test_delete_docs_empty_ids(self):
        self.client.create_index(index_name=self.index_name_1)
        self.client.index(self.index_name_1).add_documents([{"abc": "efg", "_id": "123"}], non_tensor_fields=[])
        try:
            self.client.index(self.index_name_1).delete_documents([])
            raise AssertionError
        except MarqoWebError as e:
            assert "can't be empty" in str(e) or "value_error.missing" in str (e)
        res = self.client.index(self.index_name_1).get_document("123")
        print(res)
        assert "abc" in res