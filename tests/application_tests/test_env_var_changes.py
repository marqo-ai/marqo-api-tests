"""

To test these functions locally:

1. Run a Marqo container another terminal (these tests assume there is a running Marqo
    container and then try to kill it, failing if unsuccessful)
2. cd into the root of this repo
3. Run the following command (you can replace MARQO_IMAGE_NAME):

    TESTING_CONFIGURATION=DIND_MARQO_OS \
    MARQO_API_TESTS_ROOT=. \
    MARQO_IMAGE_NAME=marqoai/marqo:latest \
    pytest tests/application_tests/test_env_var_changes.py::TestEnvVarChanges::test_multiple_env_vars


We may test multiple different env vars in the same test case. This is because
 each new test case is expensive, requiring a restart of Marqo. This prevents
 this test suite's runtime from growing too large.
"""
from tests import marqo_test
from tests import utilities
from marqo import Client
from marqo.errors import MarqoApiError, MarqoWebError
import json
import os
import time
from datetime import datetime


class TestEnvVarChanges(marqo_test.MarqoTestCase):

    """
        All tests that rerun marqo with different env vars should go here
        Teardown will handle resetting marqo back to base settings
    """

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def tearDown(self) -> None:
        utilities.control_marqo_os("marqo-os", "start")
    
    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        # Ensures that marqo goes back to default state after these tests
        utilities.rerun_marqo_with_default_config(
            calling_class=cls.__name__
        )
        print("Marqo has been rerun with default env vars!")

    def test_max_replicas(self):
        """
        Tests Marqo with a different max replica count to default (Default max is 1).

        Also, tests that INFO-level log output is as expected.
        """
        max_replicas = 5
        log_level = 'info'
        print(f"Attempting to rerun marqo with max replicas: {max_replicas} and log level {log_level}")
        utilities.rerun_marqo_with_env_vars(
            env_vars=[
                "-e", f"MARQO_MAX_NUMBER_OF_REPLICAS={max_replicas}",
                "-e", f"MARQO_LOG_LEVEL={log_level}"],
            calling_class=self.__class__.__name__
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

        # ## Testing log output when LEVEL=info ##
        #    we want to ensure that no excessive log messages are printed
        log_blob = utilities.retrieve_docker_logs(container_name='marqo')
        assert ("Unverified HTTPS request is being made to host 'host.docker.internal'. "
                "Adding certificate verification is strongly advised." not in log_blob)

        assert ("Unverified HTTPS request is being made to host 'localhost'. "
                "Adding certificate verification is strongly advised." not in log_blob)
        assert 'torch==' not in log_blob
        assert 'Status: Downloaded newer image for marqoai/marqo-os' not in log_blob
        assert 'FutureWarning: Importing `GenerationMixin`' not in log_blob
        assert 'Called redis-server command' not in log_blob

        # to assure use that logs aren't just completely empty:
        assert 'COMPLETED SUCCESSFULLY' in log_blob
        assert 'Marqo throttling successfully started.' in log_blob
        assert 'INFO:DeviceSummary:found devices' in log_blob
        assert 'INFO:ModelsForStartup:completed loading models' in log_blob
        # ## End log output tests ##

    def test_preload_models(self):
        """
        Tests rerunning marqo with non-default, custom model.
        Default models are ["hf/all_datasets_v4_MiniLM-L6", "ViT-L/14"]

        Also, this tests log output when log level is not set. The log level should be INFO by default.
        """

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
            env_vars = ['-e', f"MARQO_MODELS_TO_PRELOAD=[{json.dumps(open_clip_model_object)}]"],
            calling_class=self.__class__.__name__
        )

        # check preloaded models (should be custom model)
        custom_models = ["open-clip-1"]
        res = self.client.get_loaded_models()
        assert set([item["model_name"] for item in res["models"]]) == set(custom_models)

        # ## Testing log output when log level is not set. ##
        #    we want to ensure that no excessive log messages are printed
        log_blob = utilities.retrieve_docker_logs(container_name='marqo')
        assert ("Unverified HTTPS request is being made to host 'host.docker.internal'. "
                "Adding certificate verification is strongly advised." not in log_blob)

        assert  ("Unverified HTTPS request is being made to host 'localhost'. "
                 "Adding certificate verification is strongly advised." not in log_blob)
        assert 'torch==' not in log_blob
        assert 'Status: Downloaded newer image for marqoai/marqo-os' not in log_blob
        assert  'FutureWarning: Importing `GenerationMixin`' not in log_blob
        assert 'Called redis-server command' not in log_blob

        # to assure use that logs aren't just completely empty:
        assert 'COMPLETED SUCCESSFULLY' in log_blob
        assert 'Marqo throttling successfully started.' in log_blob
        assert 'INFO:DeviceSummary:found devices' in log_blob
        assert 'INFO:ModelsForStartup:completed loading models' in log_blob
        # ## End log output tests ##

    def test_multiple_env_vars(self):
        """
            Ensures that rerun_marqo_with_env_vars can work with several different env vars
            at the same time

            3 things in the same command:
            1. Load models
            2. set max number of replicas
            3. set max EF
            4. set log level to debug

            Also, asserts that Marqo's debug output is as expected
        """

        # Restart marqo with new max values
        max_replicas = 10
        max_ef = 6000
        new_models = ["hf/all_datasets_v4_MiniLM-L6"]
        utilities.rerun_marqo_with_env_vars(
            env_vars=[
                "-e", f"MARQO_MAX_NUMBER_OF_REPLICAS={max_replicas}",
                "-e", f"MARQO_EF_CONSTRUCTION_MAX_VALUE={max_ef}",
                "-e", f"MARQO_MODELS_TO_PRELOAD={json.dumps(new_models)}",
                "-e", f"MARQO_LOG_LEVEL=debug"
            ],
            calling_class=self.__class__.__name__
        )

        # Create index with same number of replicas and EF
        res_0 = self.client.create_index(index_name=self.index_name_1, settings_dict={
            "number_of_replicas": 4,  # should be fine now
            "index_defaults": {
                "ann_parameters": {
                    "space_type": "cosinesimil",
                    "parameters": {
                        "ef_construction": 5000,  # should be fine now
                        "m": 16
                    }
                }
            }
        })

        # Assert correct replicas
        # Make sure new index has 4 replicas
        assert self.client.get_index(self.index_name_1).get_settings() \
                   ["number_of_replicas"] == 4

        # Assert correct EF const
        assert self.client.get_index(self.index_name_1).get_settings() \
                   ["index_defaults"]["ann_parameters"]["parameters"]["ef_construction"] == 5000

        # Assert correct models
        res = self.client.get_loaded_models()
        assert set([item["model_name"] for item in res["models"]]) == set(new_models)

        # ## Testing log output when LEVEL=debug ##
        #    we want to ensure that  in debug mode, no information is hidden

        # use the index to generate more log outputs (specifically regarding HTTPS requests)
        self.client.index(self.index_name_1).add_documents(
            documents=[{'Title': 'Recipes for hippos'}],
            tensor_fields=['Title'],
            auto_refresh=True
        )
        self.client.index(self.index_name_1).search('something')
        log_blob = utilities.retrieve_docker_logs(container_name='marqo')
        assert 'torch==' in log_blob
        # assertions about HTTP requests aren't reliable for some reason
        # assert ((
        #                 "Unverified HTTPS request is being made to host 'host.docker.internal'. "
        #                 "Adding certificate verification is strongly advised." in log_blob)
        #         or (
        #                 "Unverified HTTPS request is being made to host 'localhost'. "
        #                 "Adding certificate verification is strongly advised." in log_blob))
        if "DIND" in os.environ["TESTING_CONFIGURATION"]:
            assert 'Status: Downloaded newer image for marqoai/marqo-os' in log_blob
        assert 'FutureWarning: Importing `GenerationMixin`' in log_blob
        assert 'Called redis-server command' in log_blob

    def test_max_add_docs_count(self):
        """
        Test that MARQO_MAX_ADD_DOCS_COUNT works as expected. Trying to add more documents than the limit should fail.
        """

        counts_to_test = [10, 50, 100]
        for count in counts_to_test:
            # Restart marqo with new max values
            utilities.rerun_marqo_with_env_vars(
                env_vars = [
                    "-e", f"MARQO_MAX_ADD_DOCS_COUNT={count}",
                ],
                calling_class=self.__class__.__name__
            )

            # Create the index
            self.client.create_index(index_name=self.index_name_1)

            # Add 1 less document than the maximum
            self.client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"} for _ in range(count-1)
            ], device="cpu", non_tensor_fields=[])

            # Add exactly the maximum number of docs
            self.client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"} for _ in range(count)
            ], device="cpu", non_tensor_fields=[])

            # Add more than the maximum but BATCHED (should succeed)
            self.client.index(self.index_name_1).add_documents(documents=[
                {"d1": "blah"} for _ in range(count+1)
            ], device="cpu", non_tensor_fields=[], client_batch_size=count//2)

            # Add more than the maximum (should fail with bad request)
            try:
                self.client.index(self.index_name_1).add_documents(documents=[
                    {"d1": "blah"} for _ in range(count+1)
                ], device="cpu", non_tensor_fields=[])
                raise AssertionError("Add docs call should have failed with bad request")
            except MarqoWebError as e:
                assert e.code == "bad_request"

    def test_marqo_default_retry_multi_env_values(self):
        """
            Ensures that retries are implemented due to transient
            network errors occuring when sending search requests to OpenSearch.
        """

        retry_attempt_list = [1,3,5,7,10]

        for retry_attempt in retry_attempt_list:
            utilities.rerun_marqo_with_env_vars(
                env_vars=[
                    "-e", f"DEFAULT_MARQO_MAX_BACKEND_RETRY_ATTEMPTS='{retry_attempt}'"
                ],
                calling_class=self.__class__.__name__
            )
            self.client.create_index(self.index_name_1)
            
            for search_method in ["TENSOR", "LEXICAL"]:
                utilities.control_marqo_os("marqo-os", "stop")
                try:
                    start_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%dT%H:%M:%S")
                    res = self.client.index(self.index_name_1).search(
                        q="blah",
                        device="cpu",
                        search_method=search_method
                    )
                except Exception as e:
                    assert e.__class__ == MarqoWebError
                    pass

                log_blob = utilities.retrieve_docker_logs("marqo", start_time)

                retry_text = "BackendCommunicationError encountered... Retrying request to"
                assert retry_text in log_blob
                assert log_blob.count(retry_text) == retry_attempt

                utilities.control_marqo_os("marqo-os", "start")

    def test_marqo_add_docs_retry_multi_env_values(self):
        """
            Ensures that retries are implemented due to transient
            network errors occuring when sending add docs requests to OpenSearch.
        """
        retry_attempt_list = [1,3,5,7,10]

        for retry_attempt in retry_attempt_list:
            utilities.rerun_marqo_with_env_vars(
                env_vars=[
                    "-e", f"MARQO_MAX_BACKEND_ADD_DOCS_RETRY_ATTEMPTS='{retry_attempt}'"
                ],
                calling_class=self.__class__.__name__
            )

            self.client.create_index(self.index_name_1)

            res = self.client.index(self.index_name_1).add_documents(
                documents=[{"some": "data"}, {"some1": "data1"}],
                tensor_fields=["some", "some1"]
            ) # add docs to populate index cache
            self.client.index(self.index_name_1).refresh()

            res = self.client.index(self.index_name_1).search(
                q="blah",
                device="cpu",
                search_method="TENSOR"
            )

            utilities.control_marqo_os("marqo-os", "stop")

            try:
                start_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%dT%H:%M:%S")
                res = self.client.index(self.index_name_1).add_documents(
                    documents=[{"some2": "data2"}, {"some3": "data3"}],
                    tensor_fields=["some2", "some3"]
                )
            except Exception as e:
                assert e.__class__ == MarqoWebError
                pass

            log_blob = utilities.retrieve_docker_logs("marqo", start_time)

            retry_text = "BackendCommunicationError encountered... Retrying request to"
            assert retry_text in log_blob
            assert log_blob.count(retry_text) == retry_attempt

            utilities.control_marqo_os("marqo-os", "start")

    def test_marqo_search_retry_multi_env_values(self):
        """
            Ensures that retries are implemented due to transient
            network errors occuring when sending search requests to OpenSearch.
        """

        retry_attempt_list = [1,3,5,7,10]

        for retry_attempt in retry_attempt_list:
            utilities.rerun_marqo_with_env_vars(
                env_vars=[
                    "-e", f"MARQO_MAX_BACKEND_SEARCH_RETRY_ATTEMPTS='{retry_attempt}'"
                ],
                calling_class=self.__class__.__name__
            )
            self.client.create_index(self.index_name_1)
            
            for search_method in ["TENSOR", "LEXICAL"]:
                utilities.control_marqo_os("marqo-os", "stop")
                try:
                    start_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%dT%H:%M:%S")
                    res = self.client.index(self.index_name_1).search(
                        q="blah",
                        device="cpu",
                        search_method=search_method
                    )
                except Exception as e:
                    assert e.__class__ == MarqoWebError
                    pass

                log_blob = utilities.retrieve_docker_logs("marqo", start_time)

                retry_text = "BackendCommunicationError encountered... Retrying request to"
                assert retry_text in log_blob
                assert log_blob.count(retry_text) == retry_attempt

                utilities.control_marqo_os("marqo-os", "start")

