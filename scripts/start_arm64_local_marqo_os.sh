#!/bin/bash
# args:
# $1 : marqo_image_name - name of the image you want to test
# $@ : env_vars - strings representing all args to pass docker call
set -x
export LOCAL_OPENSEARCH_URL="https://localhost:9200"

docker rm -f marqo-os
docker run --name marqo-os -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3-arm &

# wait for marqo-os to start
# Start the timer
start=$(date +%s.%N)
until [[ $(curl -v --silent --insecure $LOCAL_OPENSEARCH_URL 2>&1 | tee /dev/tty | grep Unauthorized) ]]; do
    sleep 0.1;
done;
# End the timer
end=$(date +%s.%N)

# Calculate the elapsed time
elapsed=$(echo "$end - $start" | bc)

# Print the elapsed time
echo "Elapsed time for marqo-os to start: $elapsed seconds"

MARQO_DOCKER_IMAGE="$1"
shift

docker rm -f marqo

# Explanation:
# -d detaches docker from process (so subprocess does not wait for it)
# ${@:+"$@"} adds ALL args (past $1) if any exist.

set -x
docker run -d --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e "MARQO_MAX_CPU_MODEL_MEMORY=1.6" \
    -e "OPENSEARCH_URL=$LOCAL_OPENSEARCH_URL" \
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