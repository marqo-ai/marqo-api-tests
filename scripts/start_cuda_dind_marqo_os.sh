#!/bin/bash
# args:
# $1 : marqo_image_name - name of the image you want to test
# $@ : env_vars - strings representing all args to pass docker call
docker rm -f marqo;

MARQO_DOCKER_IMAGE="$1"
shift

# Explanation:
# -d detaches docker from process (so subprocess does not wait for it)
# ${@:+"$@"} adds ALL args (past $1) if any exist.
set -x
docker run -d --name marqo --gpus all --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway \
  -e MARQO_ENABLE_BATCH_APIS=TRUE \
  -e MARQO_MODELS_TO_PRELOAD=[] \
    ${@:+"$@"} "$MARQO_DOCKER_IMAGE"
set +x

# Follow docker logs (since it is detached)
docker logs -f marqo &
LOGS_PID=$!

# wait for marqo to start
until [[ $(curl -v --silent --insecure http://localhost:8882 2>&1 | grep Marqo) ]]; do
    sleep 0.1;
done;

# Kill the `docker logs` command (so subprocess does not wait for it)
kill $LOGS_PID