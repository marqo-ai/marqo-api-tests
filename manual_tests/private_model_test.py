"""

Todo: test mp
"""
import os
import marqo.errors
import marqo
from dotenv import load_dotenv

import datetime

print('now it is', datetime.datetime.now())

DEVICE='cuda'
# if TEST_SEARCH, then search() is used to load the model. Else add_docs is used
TEST_SEARCH = True

load_dotenv()

acc_key = os.environ['s3_acc_key']
sec_acc_key = os.environ['s3_sec_acc_key']
hf_token = os.environ['hf_token']

bucket = 'model-authentication-test'
ob = 'dummy customer/vit_b_32-quickgelu-laion400m_e31-d867053b.pt'

hf_repo_name = "Marqo/test-private"
hf_object = "dummy_model.pt"


index_name = 'index_name'


mq = marqo.Client(**{
    "url": os.environ['MARQO_URL'],
    'api_key': os.environ['MARQO_API_KEY']
})

def _get_base_index_settings():
    return {
        "index_defaults": {
            "treat_urls_and_pointers_as_images": True,
            "model": 'my_model2',
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


def _get_hf_settings():
    ix_settings = _get_base_index_settings()
    ix_settings['index_defaults']['model_properties'] = {
        "name": "ViT-B/32",
        "dimensions": 512,
        "model_location": {
            "hf": {
                "repo_id": hf_repo_name,
                "filename": hf_object,
            },
            "auth_required": True
        },
        "type": "open_clip",
    }
    return ix_settings


def clean_up():
    try:
        mq.delete_index(index_name=index_name)
    except marqo.errors.MarqoWebError:
        pass

def run_custom_model_test():
    """add docs -> search"""

    # use hf auth during search/add docs. Else s3 auth will be used
    TESTING_HF = False

    same_arch_settings  = {"index_defaults": {
        "treat_urls_and_pointers_as_images": True,
        "model": "ViT-B/32"
    }}
    l14_settings  = {"index_defaults": {
        "treat_urls_and_pointers_as_images": True,
        "model": "ViT-L/14"
    }}


    # -------------

    if TESTING_HF:
        model_auth = {'hf': {'token': hf_token}}
        settings = _get_hf_settings()
    else:
        model_auth = {'s3': {"aws_access_key_id": acc_key, "aws_secret_access_key": sec_acc_key}}
        settings = _get_s3_settings()

    try:
        mq.create_index(
            index_name=index_name, settings_dict=settings,
        )
        print('index settings: \n', mq.index(index_name=index_name).get_settings(), '\n')
    except Exception as e:
        print(f'        Error during create index: {e}')
    if TEST_SEARCH:
        print('starting with search')
        print(
            mq.index(index_name=index_name).search(
                device=DEVICE,
                q="Hehehe",
                model_auth=model_auth
            )
        )
    print('adding sample docs')
    print(
        mq.index(index_name=index_name).add_documents(
            auto_refresh=True, documents=[{'a': 'b', "_id": "ok"}],
            model_auth=model_auth,
            device=DEVICE
        )
    )
    print(
        mq.index(index_name=index_name).add_documents(
            auto_refresh=True, documents=[{'a': 'b', "_id": "ok_2"}],
            model_auth=model_auth,
            device=DEVICE
        )
    )
    doc = mq.index(index_name=index_name).get_document(document_id='ok', expose_facets=True)
    doc2 = mq.index(index_name=index_name).get_document(document_id='ok_2', expose_facets=True)
    assert doc['_tensor_facets'][0]['_embedding'][:5] == doc2['_tensor_facets'][0]['_embedding'][:5]
    return doc['_tensor_facets'][0]['_embedding'][:5]


def run_hf_test():
    """add docs -> search"""
    mq.create_index(
        index_name=index_name, settings_dict=_get_hf_settings(),
    )
    ar =  mq.index(index_name=index_name).add_documents(
        auto_refresh=True, documents=[
                {'title': f'rock {i} bread', '_id': f'id_{i}'} for i in range(20)
            ],
        model_auth={'hf': {'token': hf_token}},
        processes=4, server_batch_size=5,
        device=DEVICE
    )
    print(ar)
    sr = mq.index(index_name=index_name).search(
        q="what is healthy fruit I can eat",
        model_auth={'hf': {'token': hf_token}},
        device=DEVICE
    )
    print(sr)

def evaluate_vector(truncated_new_vec):
    """Compares the first 5 elements of a new vector with the default vectors"""
    samples = get_default_vectors()
    print('comparing new vector with samples. First vector is the new one')
    print('x', truncated_new_vec)
    for i, k in enumerate(samples):
        if not k.startswith('target_vec'):
            assert len(truncated_new_vec) == len(samples[k])
            assert truncated_new_vec != samples[k]
            print (i, samples[k])
    if DEVICE.lower() == 'cuda':
        assert truncated_new_vec == samples['target_vec_cuda']
    else:
        assert truncated_new_vec == samples['target_vec_cpu']
    print('Asserted', "truncated_new_vec == samples['target_vec'] is True")


def get_default_vectors():
    return {
        '{}': [-0.036825817078351974, 0.023316672071814537, -0.06834684312343597, 0.026685357093811035, -0.045200563967227936],
        "{'index_defaults': {'treat_urls_and_pointers_as_images': True, 'model': 'ViT-B/32'}}": [0.011017965152859688, 0.018727311864495277, 0.040575817227363586, 0.018001120537519455, -0.013616213575005531],
        "{'index_defaults': {'treat_urls_and_pointers_as_images': True, 'model': 'ViT-L/14'}}": [0.004505092278122902, 0.009140913374722004, 0.0325147807598114, -0.01702379807829857, 0.005428432021290064],
        'target_vec_cpu' : [-0.019395295530557632, 0.0367489829659462, 0.009825248271226883, -0.004976424388587475, 0.006592699326574802],
        'target_vec_cuda': [-0.01889081671833992, 0.036698516458272934, 0.009848922491073608, -0.00475456053391099, 0.007050872314721346]
    }
clean_up()
print('loaded models, before testing', mq.get_loaded_models())
print()
evaluate_vector(run_custom_model_test())
clean_up()

for m in mq.get_loaded_models()['models']:
    print('ejecting', m['model_name'])
    mq.eject_model(model_name=m['model_name'], model_device=m['model_device'])


