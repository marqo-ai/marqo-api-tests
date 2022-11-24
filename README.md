# API test suite for Marqo

## Prerequisites
- Have Docker installed
- Be able to run docker [without sudo](https://github.com/sindresorhus/guides/blob/main/docker-without-sudo.md). 
You may need to run `newgrp docker` after the instructions in the link
if you are getting still getting a `permission denied` error.
- Have git clone access to the Marqo repo (everyone should as it's public)
- Have Python3.8 installed
- Have pip installed
- Have the requirements found in `requirements.txt` installed
- _For Arm64 Machines (ignore if you have an amd64/intel chip)_:
    - Install QEMU (this allows you to emulate the x86 instruction set on the ARM processor, needed for marqo-os). 

# !!! Warning !!!
__These integration tests (especially when run with `tox`) will mutate clusters it has access to. 
Tox also runs scripts which remove Marqo, Marqo-os, OpenSearch containers and build images.__

__It is recommended for the full tests suite to run on a machine with lots of storage but no access to prod instances.__
## Set up

1. Clone this repo
2. Make a copy of `conf_sample` called `conf` in the same directory. 
Fill in the environment variables/credentials in `conf` as appropriate. 
The `conf` file will be read by the startup scripts in order to populate environment variables.

## Run the test suite locally

1. Have Marqo instance running
2. Export the `TESTING_CONFIGURATION` variable to `CUSTOM`. This tells the integration tests what configuration
is currently being tested. Enter this command in a terminal:
   - `export TESTING_CONFIGURATION="CUSTOM"`
3. Then, in the same terminal, run `pytest tests/`

### To run a specific test:
4. Export the test dir to `PYTHONPATH` by running the following: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/tests"`
5. Run pytest and specify the desired test/subdir. For example: 
 
```
pytest tests/api_tests/test_neural_search.py::TestAddDocuments::test_prefiltering
```

## Run all tests (in multi configurations)
This runs the tox tests including startup and cleanup scripts. This can mutate clusters it has access to.

This removes Marqo and Marqo-os containers found on the machine and will build Marqo images from the cloned repo. 

`cd` into the api testing repo home directory and run `tox`, to test all environments. 

To run a specific environment do `tox -e <TOX ENVIRONMENT NAME>`



Run the following  if you are on Ubuntu:
1. `sudo apt-get install -y qemu-user-static`
## Devloping
If you are going to make a new test environment, make sure you set the `TESTING_CONFIGURATION` environment variable so
that the test suite knows if whether or not to modify certain tests for the current configuration 

You do this by creating a `setenv` section in `tox.ini`:
```tox
setenv =
  TESTING_CONFIGURATION = YOUR_TEST_ENV_NAME
```
### Future work
* Have a tox var to specify the image name. This allows for remote images to be tested, in addition to local builds `marqo_image_name = marqo_docker_0`
