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
import json

import pytest
from marqo.errors import MarqoWebError

from tests import marqo_test
from tests import utilities


@pytest.mark.fixed
class TestEnvVarChanges(marqo_test.MarqoTestCase):

    """
        All tests that rerun marqo with different env vars should go here
        Teardown will handle resetting marqo back to base settings
    """
    
    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        # Ensures that marqo goes back to default state after these tests
        utilities.rerun_marqo_with_default_config(
            calling_class=cls.__name__
        )
        print("Marqo has been rerun with default env vars!")

    def test_preload_models(self):
        """
        Tests rerunning marqo with non-default, custom model.
        Default models are ["hf/all_datasets_v4_MiniLM-L6", "ViT-L/14"]

        Also, this tests log output when log level is not set. The log level should be INFO by default.
        """

        open_clip_model_object = {
            "model": "open-clip-1",
            "modelProperties": {
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
        self.client.create_index("test_index_for_preloaded_models")
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
            2. set max EF
            3. set log level to debug

            Also, asserts that Marqo's debug output is as expected
        """

        # Restart marqo with new max values
        max_ef = 6000
        new_models = ["hf/all_datasets_v4_MiniLM-L6"]
        index_name = "test_multiple_env_vars"
        utilities.rerun_marqo_with_env_vars(
            env_vars=[
                "-e", f"MARQO_EF_CONSTRUCTION_MAX_VALUE={max_ef}",
                "-e", f"MARQO_MODELS_TO_PRELOAD={json.dumps(new_models)}",
                "-e", f"MARQO_LOG_LEVEL=debug"
            ],
            calling_class=self.__class__.__name__
        )

        # Create index with same number of replicas and EF
        res_0 = self.client.create_index(index_name=index_name, ann_parameters={
            "spaceType": 'prenormalized-angular', "parameters": {"efConstruction": 5000, "m": 16}}
        )

        # Assert correct EF const
        assert self.client.get_index(index_name).get_settings() \
                   ["annParameters"]["parameters"]["efConstruction"] == 5000

        # Assert correct models
        res = self.client.get_loaded_models()
        assert set([item["model_name"] for item in res["models"]]) == set(new_models)

        # ## Testing log output when LEVEL=debug ##
        #    we want to ensure that  in debug mode, no information is hidden

        # use the index to generate more log outputs (specifically regarding HTTPS requests)
        self.client.index(index_name).add_documents(
            documents=[{'Title': 'Recipes for hippos'}],
            tensor_fields=['Title'],
        )
        self.client.index(index_name).search('something')
        log_blob = utilities.retrieve_docker_logs(container_name='marqo')
        assert 'torch==' in log_blob

    def test_max_add_docs_count(self):
        """
        Test that MARQO_MAX_ADD_DOCS_COUNT works as expected. Trying to add more documents than the limit should fail.
        """

        counts_to_test = [10, 50, 100]
        index_name = "test_max_add_docs_count"
        for count in counts_to_test:
            # Restart marqo with new max values
            utilities.rerun_marqo_with_env_vars(
                env_vars = [
                    "-e", f"MARQO_MAX_ADD_DOCS_COUNT={count}",
                ],
                calling_class=self.__class__.__name__
            )

            # Create the index
            self.client.create_index(index_name=index_name)

            # Add 1 less document than the maximum
            self.client.index(index_name).add_documents(documents=[
                {"d1": "blah"} for _ in range(count-1)
            ], device="cpu", tensor_fields=["d1"])

            # Add exactly the maximum number of docs
            self.client.index(index_name).add_documents(documents=[
                {"d1": "blah"} for _ in range(count)
            ], device="cpu", tensor_fields=["d1"])

            # Add more than the maximum but BATCHED (should succeed)
            self.client.index(index_name).add_documents(documents=[
                {"d1": "blah"} for _ in range(count+1)
            ], device="cpu", tensor_fields=["d1"], client_batch_size=count//2)

            # Add more than the maximum (should fail with bad request)
            try:
                self.client.index(index_name).add_documents(documents=[
                    {"d1": "blah"} for _ in range(count+1)
                ], device="cpu", tensor_fields=["d1"])
                raise AssertionError("Add docs call should have failed with bad request")
            except MarqoWebError as e:
                assert e.code == "bad_request"