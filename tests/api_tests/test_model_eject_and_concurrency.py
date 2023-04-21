from marqo.client import Client
from marqo.errors import MarqoWebError
from tests.marqo_test import MarqoTestCase
import time
import threading, queue, multiprocessing


class TestModelEjectAndConcurrency(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.client = Client(**cls.client_settings)
        cls.index_model_object = {
            "test_0": 'open_clip/ViT-B-32/laion400m_e31',
            "test_1": 'open_clip/ViT-B-32/laion400m_e32',
            "test_2": 'open_clip/RN50x4/openai',
            "test_3": 'open_clip/RN101-quickgelu/yfcc15m',
            "test_4": 'open_clip/ViT-B-16-plus-240/laion400m_e32',
            "test_5": 'open_clip/ViT-B-32-quickgelu/laion400m_e31',
            "test_6": "hf/all-MiniLM-L6-v1",
            "test_7": "hf/all-MiniLM-L6-v2",
            "test_8": "hf/all_datasets_v3_MiniLM-L12",
            "test_9": 'open_clip/ViT-B-32/laion2b_e16',
            "test_10": 'ViT-B/16',
            "test_11": 'open_clip/convnext_base/laion400m_s13b_b51k',
            "test_12": 'open_clip/ViT-B-16/laion400m_e32',
            "test_13": 'open_clip/ViT-B-16/laion2b_s34b_b88k',
        }

        for index_name, model in cls.index_model_object.items():
            settings = {
                "model": model
            }
            try:
                cls.client.delete_index(index_name)
            except Exception:
                pass

            cls.client.create_index(index_name, **settings)

            cls.client.index(index_name).add_documents([
                {
                    "Title": "The Travels of Marco Polo",
                    "Description": "A 13th-century travelogue describing Polo's travels"
                },
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection, "
                                   "mobility, life support, and communications for astronauts",
                    "_id": "article_591"
                }], auto_refresh=True)

        time.sleep(10)

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)

    def tearDown(self) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        for index_name, model in cls.index_model_object.items():
            try:
                cls.client.delete_index(index_name)
            except Exception:
                pass

    def normal_search(self, index_name, q):
        # A function will be called in multiprocess
        res = self.client.index(index_name).search("what is best to wear on the moon?")
        if len(res["hits"]) == 2:
            q.put("normal search success")
        else:
            q.put(AssertionError)

    def racing_search(self, index_name, q):
        # A function will be called in multiprocess
        try:
            res = self.client.index(index_name).search("what is best to wear on the moon?")
            q.put(AssertionError)
        except MarqoWebError as e:
            if "Request rejected, as this request attempted to update the model cache," in str(e):
                q.put("racing search get blocked with correct error")
            else:
                q.put(e)

    def test_sequentially_search(self):
        for index_name in list(self.index_model_object):
            self.client.index(index_name).search(q='What is the best outfit to wear on the moon?')

    def test_concurrent_search_with_cache(self):
        # Search once to make sure the model is in cache
        test_index = "test_1"
        res = self.client.index(test_index).search("what is best to wear on the moon?")

        normal_search_queue = queue.Queue()
        threads = []
        for i in range(2):
            t = threading.Thread(target=self.normal_search, args=(test_index, normal_search_queue))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert normal_search_queue.qsize() == 2
        while not normal_search_queue.empty():
            assert normal_search_queue.get() == "normal search success"

    def test_concurrent_search_without_cache(self):
        # Remove all the cached models
        super().removeAllModels()

        test_index = "test_10"
        normal_search_queue = queue.Queue()
        racing_search_queue = queue.Queue()
        threads = []
        main_thread = threading.Thread(target=self.normal_search, args=(test_index, normal_search_queue))
        main_thread.start()
        time.sleep(0.2)

        for i in range(2):
            t = threading.Thread(target=self.racing_search, args=(test_index, racing_search_queue))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        main_thread.join()

        assert normal_search_queue.qsize() == 1
        while not normal_search_queue.empty():
            assert normal_search_queue.get() == "normal search success"

        assert racing_search_queue.qsize() == 2
        while not racing_search_queue.empty():
            assert racing_search_queue.get() == "racing search get blocked with correct error"


