#!/bin/bash
# args:
# $1 : marqo_image_name - name of the image you want to test
# $2 : env_vars - string representing the env vars to pass marqo
set -x
docker rm -f marqo;

# docker call args array
# declare -a args
#args=("--name" "marqo" "--gpus" "all" "--privileged" "-p" "8882:8882" "--add-host" "host.docker.internal:host-gateway")

# add $2 env vars if it's set
#if [ -n "$2" ]; then
#    echo "checking arg2: $2"
#    # split $2 into an array on spaces
#    IFS=' ' read -r -a env_vars <<< "$2"
#    # add each item in env_vars to args
#    for var in "${env_vars[@]}"; do
#        args+=("$var")
#    done
#fi

# add $1 marqo image name to args
#args+=("$1")
#docker run -d "${args[@]}"

# TODO put this back (remove models to preload)
docker run -d --name marqo --gpus all --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway ${2:+$2} "$1"

# Follow the logs of the Docker container in the background and capture its PID
docker logs -f marqo &
LOGS_PID=$!

# wait for marqo to start
set +x
until [[ $(curl -v --silent --insecure http://localhost:8882 2>&1 | grep Marqo) ]]; do
    sleep 0.1;
done;

# Kill the `docker logs` command (so subprocess does not wait for it)
kill $LOGS_PID