"""This file contains two test suites:
    - TestFiltering which tests filtering on content with special characters
    - TestFieldnames which tests fieldnames with special characters.

TestFieldNames tests the following:
    - Identifies and tests which special characters can be indexed
    - Identifies and tests which special characters work in fieldnames used as searchable attributes
    - Identifies and tests which special characters work in fieldnames used as within filters
"""
import uuid
from typing import List

import pytest
from marqo.client import Client
from marqo.errors import MarqoApiError

from tests.marqo_test import MarqoTestCase


@pytest.mark.fixed
class TestFiltering(MarqoTestCase):
    """More rigorous tests for filtering with special characters

    We don't include tests for spaces, as these are tested elsewhere
    """
    supported_special_str_sequences = {
        '/', '*', '^', '\\', '!', '[', '||', '?',
        '&&', '"', ']', '-', '{', '~', '+', '}', ':', ')', '(', '.', '\n', '\t', '\r',
    }

    unsupported_special_str_sequences = {
        '\b', '\f', '\v', '\a',
    }

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.standard_structured_index_name = "structured_standard" + str(uuid.uuid4()).replace('-', '')
        cls.standard_unstructured_index_name = "unstructured_standard" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.standard_unstructured_index_name,
                "type": "unstructured"
            },
            {
                "indexName": cls.standard_structured_index_name,
                "type": "structured",
                "allFields": [{"name": "searchField", "type": "text", "features":["lexical_search"]},
                              {"name": "filteringField", "type": "text", "features": ["filter"]}],
                "tensorFields": ["searchField"]
            }
        ])

        cls.indexes_to_delete = [cls.standard_unstructured_index_name, cls.standard_structured_index_name]

    def test_filtering_on_content_with_special_chars_tensor(self):
        for index_name in [self.standard_structured_index_name, self.standard_unstructured_index_name]:
            for special_str in self.supported_special_str_sequences:
                docs = [
                    {'_id': 'doc_0', 'filteringField': f"str_at_back_{special_str}", "searchField": "hello"},
                    {'_id': 'doc_1', 'filteringField': f"{special_str}_str_at_front", "searchField": "hello"},
                    {'_id': 'doc_2', 'filteringField': f"str_{special_str}_mid", "searchField": "hello"},
                    {'_id': 'doc_3', 'filteringField': f"{special_str}", "searchField": "hello"},
                ]
                for expected_document in docs:
                    str_for_filtering = expected_document['filteringField'].replace(special_str, f"\\{special_str}")

                    with self.subTest(f"Tensor Search, {index_name}, "
                                      f"str for filtering {str_for_filtering}, expected_doc {expected_document}"):
                        self.clear_indexes(self.indexes_to_delete)

                        self.client.index(index_name).add_documents(docs, tensor_fields=["searchField"] if \
                            index_name.startswith("un") else None)

                        res = self.client.index(index_name).search(q="hello",
                                                                   filter_string=f"filteringField:{str_for_filtering}", )

                        self.assertEqual(res["hits"][0]["_id"], expected_document["_id"])

    def test_filtering_on_content_with_special_chars_lexical(self):
        for index_name in [self.standard_structured_index_name, self.standard_unstructured_index_name]:
            for special_str in self.supported_special_str_sequences:
                docs = [
                    {'_id': 'doc_0', 'filteringField': f"str_at_back_{special_str}", "searchField": "hello"},
                    {'_id': 'doc_1', 'filteringField': f"{special_str}_str_at_front", "searchField": "hello"},
                    {'_id': 'doc_2', 'filteringField': f"str_{special_str}_mid", "searchField": "hello"},
                    {'_id': 'doc_3', 'filteringField': f"{special_str}", "searchField": "hello"},
                ]
                for expected_document in docs:
                    str_for_filtering = expected_document['filteringField'].replace(special_str, f"\\{special_str}")

                    with self.subTest(f"Lexical search, {index_name}, str for filtering "
                                      f"{str_for_filtering}, expected_doc {expected_document}"):
                        self.clear_indexes(self.indexes_to_delete)

                        self.client.index(index_name).add_documents(docs, tensor_fields=["searchField"] if \
                            index_name.startswith("un") else None)

                        res = self.client.index(index_name).search(q="hello",
                                                                   filter_string=f"filteringField:{str_for_filtering}",
                                                                   search_method="LEXICAL")

                        self.assertEqual(res["hits"][0]["_id"], expected_document["_id"])


class TestSpecialCharsFieldNames(MarqoTestCase):
    special_str_sequences = [
        # from Lucene:
        '/', '*', '^', '\\', '!', '[', '||', '?',
        '&&', '"', ']', '-', '{', '~', '+', '}', ':', ')', '(',
        # extra sequences:
        ' ', '\n', '\t', '\r', '\b', '\f', '\v', '\a', '.', '#'
    ]

    str_sequences_that_cant_be_indexed = ['.', '\n', '/']

    str_sequences_that_pass_with_escaping = [
        '^', '!', '[', '||', '?', '&&', '"', ']', '-', '{', '~', '+', '}', ')', '(', ' '
    ]

    # these technically behave the same as any non-special char, like 'a' or 'b', but we
    # test them to remove any ambiguity
    str_sequences_that_pass_without_escaping = [
        '\x08', '\x0c', '\x0b', '\x07', '#', '|',
        'MyString', 'a'
    ]

    # these sequences can be used as a searchable attribute, but can't be used for filtering
    str_sequences_that_can_be_indexed_but_not_filtered_on = ['*', ':']

    # these sequences can't even be used as a searchable attribute
    str_sequences_that_can_be_indexed_but_not_searchable_attr = ['\t', '\\', '\r']

    def test_special_chars_that_need_escaping_in_filter(self):
        """Test field names with special chars that need escaping in the filter string.

        This test also checks that the field name can be used as a searchable attribute, without
        requiring any user-defined escaping.

        For example, for a field name like:
            "my_unusual_field?"
        This test checks that it can be used for filtering with this filter string:
            "my_unusual_field\?:(content to filter)"
        """

        for special_str in self.str_sequences_that_pass_with_escaping:

            for search_method in ['TENSOR', 'LEXICAL']:
                testing_docs = [
                    {'_id': 'doc_2', f"str_{special_str}_mid": "hippo bird"},
                    {'_id': 'doc_0', f"str_at_back_{special_str}": "hippo dog"},
                    {'_id': 'doc_1', f"{special_str}_str_at_front": "hippo fish"},

                    {'_id': 'doc_3', f"{special_str}": "hippo hello"},
                    # red herring doc
                    {'_id': 'doc_4', 'some other field': f"hippo airplane"},
                    # another testing doc
                    {'_id': 'doc_5', f"{special_str}_{special_str}_{special_str}": "hippo giraffe"},
                    {'_id': 'doc_6', f"{special_str}_{special_str}{special_str}_{special_str}": "hippo giraffe"}
                ]
                self.special_char_in_fieldname_non_escaped_filter_helper(
                    special_str=special_str,
                    search_method=search_method,
                    verbose=False,
                    all_testing_docs=testing_docs,
                    test_escaped_filter_str=True,
                    test_non_escaped_filter_str=False,
                    escape_special_str_in_filter_content=False
                )

    def test_special_chars_that_need_escaping_in_filter_and_content(self):
        """Test field names with special chars that need escaping in the filter string. This test also checks if
        the special chars can be used as the field content in the filter string.

        This test also checks that the field name can be used as a searchable attribute, without
        requiring any user-defined escaping.

        For example, for a field name like:
            "my_unusual_field?"
        This test checks that it can be used for filtering with this filter string in the field name and content:
            "my_unusual_field\?:(content to filter\?)"
        """

        for special_str in self.str_sequences_that_pass_with_escaping:
            for search_method in ['TENSOR', 'LEXICAL']:
                testing_docs = [
                    {'_id': 'doc_2', f"str_{special_str}_mid": f"hippo {special_str} bird"},
                    {'_id': 'doc_0', f"str_at_back_{special_str}": f"hippo dog {special_str}"},
                    {'_id': 'doc_1', f"{special_str}_str_at_front": f"{special_str} hippo fish"},

                    {'_id': 'doc_3', f"{special_str}": f"{special_str}hippo hello"},
                    # red herring doc
                    {'_id': 'doc_4', 'some other field': f"hippo airplane"},
                    # another testing doc
                    {'_id': 'doc_5',
                     f"{special_str}_{special_str}_{special_str}": f"{special_str}hippo {special_str}giraffe{special_str}"},
                    {'_id': 'doc_6',
                     f"{special_str}_{special_str}{special_str}_{special_str}": f"hippo giraffe{special_str}"}
                ]
                self.special_char_in_fieldname_non_escaped_filter_helper(
                    special_str=special_str,
                    search_method=search_method,
                    verbose=False,
                    all_testing_docs=testing_docs,
                    test_escaped_filter_str=True,
                    test_non_escaped_filter_str=False,
                    escape_special_str_in_filter_content=True
                )

    def test_special_chars_that_dont_need_escaping_in_filter(self):
        """Test field names with special chars that don't need escaping in the filter string.

        This is actually the case for most normal special chars, but we test them to remove ambiguity.

        This test also checks that the field name can be used as a searchable attribute, without
        requiring any user-defined escaping.

        For example, for a field name like:
            "my_unusual_field#"
        This test checks that it can be used for filtering with this filter string:
            "my_unusual_field#:(content to filter)"
        """
        for special_str in self.str_sequences_that_pass_without_escaping:
            for search_method in ['TENSOR', 'LEXICAL']:
                testing_docs = [
                    {'_id': 'doc_2', f"str_{special_str}_mid": "hippo bird"},
                    {'_id': 'doc_0', f"str_at_back_{special_str}": "hippo dog"},
                    {'_id': 'doc_1', f"{special_str}_str_at_front": "hippo fish"},

                    {'_id': 'doc_3', f"{special_str}": "hippo hello"},
                    # red herring doc
                    {'_id': 'doc_4', 'some other field': f"hippo airplane"},
                    # another testing doc
                    {'_id': 'doc_5', f"{special_str}_{special_str}_{special_str}": "hippo giraffe"},
                ]
                self.special_char_in_fieldname_non_escaped_filter_helper(
                    special_str=special_str,
                    search_method=search_method,
                    verbose=False,
                    all_testing_docs=testing_docs,
                    test_escaped_filter_str=False,
                    test_non_escaped_filter_str=True,
                    escape_special_str_in_filter_content=False
                )

    def test_special_chars_that_dont_need_escaping_in_filter_and_content(self):
        """Test field names with special chars that don't need escaping in the filter string,
        testing its presence during filtering in both in field name and content.

        This is actually the case for most normal special chars, but we test them to remove ambiguity.

        This test also checks that the field name can be used as a searchable attribute, without
        requiring any user-defined escaping.

        For example, for a field name like:
            "my_unusual_field#"
        This test checks that it can be used for filtering with this filter string as both the field name
         and content:
            "my_unusual_field#:(content to filter\#)"
        """
        for special_str in self.str_sequences_that_pass_without_escaping:
            for search_method in ['TENSOR', 'LEXICAL']:
                testing_docs = testing_docs = [
                    {'_id': 'doc_2', f"str_{special_str}_mid": f"hippo {special_str} bird"},
                    {'_id': 'doc_0', f"str_at_back_{special_str}": f"hippo dog {special_str}"},
                    {'_id': 'doc_1', f"{special_str}_str_at_front": f"{special_str} hippo fish"},

                    {'_id': 'doc_3', f"{special_str}": f"{special_str}hippo hippo hello"},
                    # red herring doc
                    {'_id': 'doc_4', 'some other field': f"hippo airplane"},
                    # another testing doc
                    {'_id': 'doc_5',
                     f"{special_str}_{special_str}_{special_str}": f"{special_str}hippo hippo {special_str}giraffe{special_str}"},
                ]
                self.special_char_in_fieldname_non_escaped_filter_helper(
                    special_str=special_str,
                    search_method=search_method,
                    verbose=True,
                    all_testing_docs=testing_docs,
                    test_escaped_filter_str=False,
                    test_non_escaped_filter_str=True,
                    escape_special_str_in_filter_content=True
                )

    def test_special_chars_that_cant_be_indexed(self):
        """These special chars can't be indexed, so the test should fail before we get to searching
        """
        for special_str in self.str_sequences_that_cant_be_indexed:
            testing_docs = [
                {'_id': 'doc_2', f"str_{special_str}_mid": "hippo bird"},
                {'_id': 'doc_0', f"str_at_back_{special_str}": "hippo dog"},
                {'_id': 'doc_1', f"{special_str}_str_at_front": "hippo fish"},

                {'_id': 'doc_3', f"{special_str}": "hippo hello"},
                # red herring doc
                {'_id': 'doc_4', 'some other field': f"hippo airplane"},
                # another testing doc
                {'_id': 'doc_5', f"{special_str}_{special_str}_{special_str}": "hippo giraffe"},
            ]
            try:
                self.special_char_in_fieldname_non_escaped_filter_helper(
                    special_str=special_str,
                    # search method doesn't matter, since we expect an error at indexing time
                    search_method='TENSOR',
                    verbose=False,
                    all_testing_docs=testing_docs,
                    test_escaped_filter_str=False,
                    test_non_escaped_filter_str=False,
                    escape_special_str_in_filter_content=False,
                )
                raise AssertionError(f"Expected AssertionError for non-indexable sequence: `{repr(special_str)}`")
            except AssertionError as e:
                assert "error adding documents" in str(e).lower()

    def test_str_sequences_that_can_be_indexed_but_not_filtered_on(self):
        """These special chars can be indexed, and can be used durning search_all and as
         searchable attributes, but can't be used in a field name in a filter string.
        """
        for special_str in self.str_sequences_that_can_be_indexed_but_not_filtered_on:
            for search_method in ['TENSOR', 'LEXICAL']:
                testing_docs = [
                    {'_id': 'doc_2', f"str_{special_str}_mid": "hippo bird"},
                    {'_id': 'doc_0', f"str_at_back_{special_str}": "hippo dog"},
                    {'_id': 'doc_1', f"{special_str}_str_at_front": "hippo fish"},

                    {'_id': 'doc_3', f"{special_str}": "hippo hello"},
                    # red herring doc
                    {'_id': 'doc_4', 'some other field': f"hippo airplane"},
                    # another testing doc
                    {'_id': 'doc_5', f"{special_str}_{special_str}_{special_str}": "hippo giraffe"},
                ]
                self.special_char_in_fieldname_non_escaped_filter_helper(
                    special_str=special_str,
                    search_method=search_method,
                    verbose=True,
                    all_testing_docs=testing_docs,
                    test_escaped_filter_str=False,
                    test_non_escaped_filter_str=False,
                    escape_special_str_in_filter_content=False
                )

    def test_str_sequences_that_can_be_indexed_but_not_as_searchable_attrib(self):
        """These special chars can be indexed, and can be used during search_all
         but can't be used as searchable attributes

        """
        for special_str in self.str_sequences_that_can_be_indexed_but_not_searchable_attr:
            for search_method in ['TENSOR', 'LEXICAL']:
                testing_docs = [
                    {'_id': 'doc_2', f"str_{special_str}_mid": "hippo bird"},
                    {'_id': 'doc_0', f"str_at_back_{special_str}": "hippo dog"},
                    {'_id': 'doc_1', f"{special_str}_str_at_front": "hippo fish"},

                    {'_id': 'doc_3', f"{special_str}": "hippo hello"},
                    # red herring doc
                    {'_id': 'doc_4', 'some other field': f"hippo airplane"},
                    # another testing doc
                    {'_id': 'doc_5', f"{special_str}_{special_str}_{special_str}": "hippo giraffe"},
                ]
                try:
                    self.special_char_in_fieldname_non_escaped_filter_helper(
                        special_str=special_str,
                        search_method=search_method,
                        verbose=True,
                        all_testing_docs=testing_docs,
                        test_escaped_filter_str=False,
                        test_non_escaped_filter_str=False,
                        escape_special_str_in_filter_content=False
                    )
                except AssertionError as e:
                    assert "expected number of searchable attribute hits" in str(e).lower()

    # HELPER METHODS

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def special_char_in_fieldname_non_escaped_filter_helper(
            self, special_str: str, search_method: str,
            all_testing_docs: List[dict],
            test_escaped_filter_str: bool, test_non_escaped_filter_str: bool,
            escape_special_str_in_filter_content: bool,
            verbose: bool = False
    ):
        """ Test for field names with special chars that do not need to be escaped in the filter string at search time.
        Args:
            special_str: the special str sequence to use in the field name
            all_testing_docs: the documents to be indexed and tested.
                Each doc should have a single field with the special char in the field name.
                Also, note that the  all_testing_docs[4] (_id=doc_4) is reserved as red herring, and is skipped
                during tests.
            search_method: the search method to use
            test_escaped_filter_str: if True, test that we can filter on the field name with the special char escaped
            test_non_escaped_filter_str: if True, test that we can filter on the field name with the special char
                not escaped
            escape_special_str_in_filter_content: if True we escape the special char in the filter content at search
                time
            verbose: whether to print out the test cases. This is useful for debugging

        Notes on behavior:
            - If a user uses a Lucene escape char in a field name, they will have to escape it if filtering at search time.
            - Users will NOT have to escape the Lucene escape chars if used in searchable_attributes
            - If a user cannot create a document because a field name contains a Lucene escape char we do not test
                using that field name within a filter string or as a searchable_attribute, as the behavior is undefined.
        """
        docs = all_testing_docs

        called_with = (
            f"\n\nspecial_str: `{repr(special_str)}`, search_method: `{search_method}`, ")
        if verbose:
            print(called_with)

        self.client.delete_index(self.index_name_1)
        self.client.create_index(self.index_name_1)
        add_docs_result = self.client.index(self.index_name_1).add_documents(
            documents=docs,
            auto_refresh=True,
            non_tensor_fields=[]
        )
        if add_docs_result['errors']:
            raise AssertionError(f"Error adding documents: {add_docs_result}, docs: {repr(docs)}")

        # does the field get searched in a general search?
        search_all_fields_result = self.client.index(self.index_name_1).search(
            q="hippo",
            search_method=search_method,
            limit=10
        )
        assert len(search_all_fields_result['hits']) == len(docs)
        assert set([hit['_id'] for hit in search_all_fields_result['hits']]) == set([d['_id'] for d in docs])
        if verbose:
            print('passed searching all fields')
        # does the field get searched while searching that specific field?
        for doc in docs:
            if verbose:
                print(f'searching with doc: `{repr(doc)}`')
            if doc['_id'] == 'doc_4':
                continue

            searchable_attrib = [k for k in doc.keys() if k != '_id'][0]
            specific_searchable_attrib_res = self.client.index(self.index_name_1).search(
                q="hippo",
                search_method=search_method,
                searchable_attributes=[searchable_attrib, ],
                limit=10
            )
            assert len(specific_searchable_attrib_res['hits']) == 1, (
                f"Didn't receive expected number of searchable attribute hits. Expected 1 hit, "
                f"got {len(specific_searchable_attrib_res['hits'])} hits. "
            )
            assert set(specific_searchable_attrib_res['hits'][0]['_id']) == set(doc['_id'])

            if verbose:
                print(f"passed searching specific attrib `{searchable_attrib}`")

            if test_escaped_filter_str:
                escaped_searchable_attrib = searchable_attrib.replace(special_str, f"\\{special_str}")
                escaped_filter_str = f"{escaped_searchable_attrib}:({doc[searchable_attrib]})"
                self.filtering_on_special_char_in_fieldname_helper(
                    filter_str=escaped_filter_str, verbose=verbose, search_method=search_method,
                    expected_doc_id=doc['_id'],
                    escape_special_str_in_filter_content=escape_special_str_in_filter_content,
                    special_str=special_str
                )
                if verbose:
                    print(
                        f"passed filtering on specific attrib `{searchable_attrib}` with escaped filter_str `{repr(escaped_filter_str)}`")

            if test_non_escaped_filter_str:
                filter_str = f"{searchable_attrib}:({doc[searchable_attrib]})"
                self.filtering_on_special_char_in_fieldname_helper(
                    filter_str=filter_str, verbose=verbose, search_method=search_method,
                    expected_doc_id=doc['_id'],
                    escape_special_str_in_filter_content=escape_special_str_in_filter_content,
                    special_str=special_str
                )
                if verbose:
                    print(
                        f"passed filtering on specific attrib `{searchable_attrib}` with non-escaped filter_str `{repr(filter_str)}`")

        self.client.delete_index(self.index_name_1)

    def filtering_on_special_char_in_fieldname_helper(
            self, filter_str: str, search_method: str, expected_doc_id: str, verbose: bool,
            escape_special_str_in_filter_content: bool, special_str: str
    ):
        """ Helper function for testing filtering on a field name with a special char in it.
        This is here to reduce code duplication between the tests for escaped and non-escaped filter strings
        """
        if verbose:
            print(f"filter_str: `{repr(filter_str)}`, {filter_str}")

        if escape_special_str_in_filter_content:
            # we escape the field content part of the filter string.
            # this algorithm won't work if the special str is a colon, but that doesn't
            # matter because we can't even filter on a field name with a colon in it

            # we do this separately from the escaping the special char in the field name
            # because whether the field name is escaped or not depends on the special char, and is
            # unreleated to filtering the same char as field content.
            escaped_filter_content = filter_str.split(":")[1][1:-1].replace(special_str, f"\\{special_str}")
            filter_str = f"{filter_str.split(':')[0]}:({escaped_filter_content})"

        specific_filterable_attrib_res = self.client.index(self.index_name_1).search(
            q="hippo",
            search_method=search_method,
            filter_string=filter_str,
            limit=10
        )

        assert len(specific_filterable_attrib_res['hits']) == 1
        assert specific_filterable_attrib_res['hits'][0]['_id'] == expected_doc_id
