from tests.utilities import disallow_environments, classwide_decorate
from tests.marqo_test import MarqoTestCase
from marqo.client import Client
from marqo.errors import MarqoApiError, MarqoError, MarqoWebError
import base64
import hmac
from hashlib import sha1

# doesn't run on local environments
# @classwide_decorate(disallow_environments, configurations=["CUSTOM"])
class TestPrivateImage(MarqoTestCase):

    def setUp(self) -> None:
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        try:
            self.client.delete_index(self.index_name_1)
        except MarqoApiError as s:
            pass

    def test_private_image(self):

        settings = {
            "treat_urls_and_pointers_as_images": True,
            "model": "ViT-B/16",
            "image_preprocessing_method": None
        }

        access_key = 'AKIA2JHYH4U23STF75PK'.encode("UTF-8")
        secret_key = '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'.encode("UTF-8")


        string_to_sign = (
            "GET\n"
            "\n"
            "\n"
            "Tue, 27 Mar 2007 19:36:42 +0000\n"
            "/awsexamplebucket1/photos/puppy.jpg"
        ).encode("UTF-8")

        signature = base64.encodebytes(
            hmac.new(
                secret_key, string_to_sign, sha1
            ).digest()
        ).strip()

        authorization = f"AWS {access_key}:{signature}"

        self.client.create_index(self.index_name_1, **settings)
        add_docs_res = self.client.index(self.index_name_1).add_documents(
            [{
                "Title": "Treatise on the future of hippos",
                "img": "https://pandu-experiments-pseudo-private.s3.us-west-2.amazonaws.com/ai_hippo_realistic.png"
            }],
            image_download_headers={"Authorization": authorization}
        )
        print(add_docs_res)


