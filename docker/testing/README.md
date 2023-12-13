# container-testing

This container is hosted on dockerhub to be used for SmartSim container
integration testing. Below are the commands to push an updated version of
the container.

## Building and interacting with container locally

```sh
# Build container
docker build -t container-testing .

# Start a shell on container to try things out
docker run -it container-testing bash
```

Within the container, you can verify that you can import packages like
smartredis or pytorch locally.

## Pushing container updates to DockerHub repository

Note: <version> is bumped each time an update is pushed.
Versions have no relation to SmartSim versions.

```sh
# See current versions to determine next version
docker image inspect --format '{{.RepoTags}}' alrigazzi/smartsim-testing

docker login

# Create tags for current build of container
docker image tag container-testing alrigazzi/smartsim-testing:latest
docker image tag container-testing alrigazzi/smartsim-testing:<version>

# Push current build of container with all tags created
docker image push --all-tags alrigazzi/smartsim-testing
```

