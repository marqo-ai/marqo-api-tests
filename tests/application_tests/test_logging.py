# test_docker_logs.py
import docker
import time
import pytest

client = docker.from_env()

def run_container(image_name):
    """Run a Docker container and return its object."""
    container = client.containers.run(image_name, detach=True)
    return container

def test_container_logs():
    # Start the container
    container = run_container("your_image_name")

    # You might want to wait for a while if your container logs take time to generate
    time.sleep(10) # Wait for 10 seconds

    # Fetch the logs
    logs = container.logs().decode("utf-8")

    # Cleanup - Stop the container and remove it
    container.stop()
    container.remove()

    # Make assertions based on logs
    assert "Expected log entry" in logs
