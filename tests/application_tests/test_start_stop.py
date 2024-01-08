import subprocess
import time

import pytest
from marqo.errors import BackendCommunicationError, MarqoWebError
from requests import HTTPError

from tests import marqo_test


@pytest.mark.fixed
class TestStartStop(marqo_test.MarqoTestCase):
    NUMBER_OF_RESTARTS = 3

    def run_start_stop(self, sig: str):
        """
        restart_number: an int which prints the restart number this
            restart represents. Helpful for debugging
        sig: Option of {'SIGTERM', 'SIGINT', 'SIGKILL'}
            the type of signal to send to the container. SIGTERM gets
            sent with the 'docker stop' command. 'SIGINT' gets sent
            by ctrl + C
        """
        # 1 retry every 10 seconds...
        NUMBER_OF_TRIES = 40
        index_name = "test_start_stop_index"
        try:
            self.client.delete_index(index_name)
        except MarqoWebError:
            pass
        d1 = {"Title": "The colour of plants", "_id": "fact_1"}
        d2 = {"Title": "some frogs", "_id": "fact_2"}
        self.client.create_index(index_name=index_name)
        self.client.index(index_name).add_documents(documents=[d1, d2], tensor_fields=["Title"])
        search_res_0 = self.client.index(index_name).search(q="General nature facts")
        assert (search_res_0["hits"][0]["_id"] == "fact_1") or (search_res_0["hits"][0]["_id"] == "fact_2")
        assert len(search_res_0["hits"]) == 2

        if sig == 'SIGTERM':
            stop_marqo_res = subprocess.run(["docker", "stop", "marqo"], check=True, capture_output=True)
            assert "marqo" in str(stop_marqo_res.stdout)
        elif sig == 'SIGINT':
            stop_marqo_res = subprocess.run(["docker", "kill", "--signal=SIGINT", "marqo"], check=True, capture_output=True)
            assert "marqo" in str(stop_marqo_res.stdout)
            time.sleep(10)
        elif sig == "SIGKILL":
            stop_marqo_res = subprocess.run(["docker", "kill", "marqo"], check=True, capture_output=True)
            assert "marqo" in str(stop_marqo_res.stdout)
            time.sleep(10)
        else:
            raise ValueError(f"bad option used for sig: {sig}. Must be one of  ('SIGTERM', 'SIGINT', 'SIGKILL')")

        try:
            self.client.index(index_name).search(q="General nature facts")
            raise AssertionError("Marqo is still accessible despite docker stopping!")
        except BackendCommunicationError as mqe:
            pass

        start_marqo_res = subprocess.run(["docker", "start", "marqo"], check=True, capture_output=True)
        assert "marqo" in str(start_marqo_res.stdout)

        for i in range(NUMBER_OF_TRIES):
            try:
                self.client.index(index_name).search(q="General nature facts")
                break
            except (MarqoWebError, HTTPError) as mqe:
                # most of the time they will be 500 errors
                # ignore too many requests response
                if isinstance(mqe, HTTPError):
                    pass
                elif not isinstance(mqe, BackendCommunicationError):
                    assert mqe.status_code == 429 or mqe.status_code == 500
                if "exceeds your S2Search free tier limit" in str(mqe):
                    raise mqe
                if i + 1 >= NUMBER_OF_TRIES:
                    raise AssertionError(f"Timeout waiting for Marqo to restart!")
                time.sleep(10)

        search_res_1 = self.client.index(index_name).search(q="General nature facts")
        assert search_res_1["hits"] == search_res_0["hits"]
        assert (search_res_1["hits"][0]["_id"] == "fact_1") or (search_res_1["hits"][0]["_id"] == "fact_2")
        return True

    def test_signal_term(self):
        for i in range(self.NUMBER_OF_RESTARTS):
            print(f"testing SIGTERM: starting restart number {i}")
            assert self.run_start_stop(sig="SIGTERM")

    def test_signal_int(self):
        for i in range(3):
            print(f"testing SIGINT: starting restart number {i}")
            assert self.run_start_stop(sig="SIGINT")

    def test_signal_kill(self):
        for i in range(3):
            print(f"testing SIGKILL: starting restart number {i}")
            assert self.run_start_stop(sig="SIGKILL")

