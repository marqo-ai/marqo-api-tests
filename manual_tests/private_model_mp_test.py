"""
The test is designed for the cloud
"""
device = 'cuda'
import os
from marqo.client import Client
from marqo.errors import MarqoApiError
import marqo.errors
import marqo
from dotenv import load_dotenv

load_dotenv()

mq = Client(**{
    "url": os.environ['MARQO_URL'],
    'api_key': os.environ['MARQO_API_KEY']
})

acc_key = os.environ['s3_acc_key']
sec_acc_key = os.environ['s3_sec_acc_key']
hf_token = os.environ['hf_token']

bucket = 'model-authentication-test'
ob = 'dummy customer/vit_b_32-quickgelu-laion400m_e31-d867053b.pt'

hf_repo_name = "Marqo/test-private"
hf_object = "dummy_model.pt"

index_name = 'index_name'

try:
    mq.delete_index(index_name)
except MarqoApiError as s:
    pass


def clean_up():
    try:
        mq.delete_index(index_name=index_name)

    except marqo.errors.MarqoWebError:
        pass


def _get_base_index_settings():
    return {
        "index_defaults": {
            "treat_urls_and_pointers_as_images": True,
            "model": 'my_model3',
            "normalize_embeddings": True,
            # notice model properties aren't here. Each test has to add it
        }
    }


def _get_s3_settings():
    ix_settings = _get_base_index_settings()
    ix_settings['index_defaults']['model_properties'] = {
        "name": "ViT-B/32",
        "dimensions": 512,
        "model_location": {
            "s3": {
                "Bucket": bucket,
                "Key": ob,
            },
            "auth_required": True
        },
        "type": "open_clip",
    }
    return ix_settings

mq.create_index(index_name)


def run_s3_test():
    """add docs -> search"""
    mq.create_index(
        index_name=index_name, settings_dict=_get_s3_settings(),
    )
    print(
        mq.index(index_name=index_name).add_documents(
            device=device,
            auto_refresh=True, documents=[
                {'title': f'rock {i} bread', '_id': f'id_{i}'} for i in range(20)
            ],
            model_auth={'s3': {"aws_access_key_id" : acc_key, "aws_secret_access_key": sec_acc_key}}
        )
    )
    print(
        mq.index(index_name=index_name).search(
            q="Hehehe", limit=108,
            device=device,
            model_auth={'s3': {"aws_access_key_id" : acc_key, "aws_secret_access_key": sec_acc_key}}
        )
    )

print(mq.get_marqo())

mods = mq.get_loaded_models()
for m in mods['models']:
    if m['model_name'] not in ['my_model3']:
        print ('ejecting', m)
        print(mq.eject_model(model_name=m['model_name'], model_device=m['model_device']))

print(f'loaded models {mods}')
clean_up()
run_s3_test()
clean_up()


