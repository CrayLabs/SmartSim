# container-testing

This container is hosted on dockerhub to be used for SmartSim container
integration testing. Below are the commands to push an updated version of
the container.

Notes:
- <org> should be replaced with the new dockerhub org, once that is created.
- <version> is bumped each time an update is pushed. Versions have no relation
             to SmartSim versions.

## Building and interacting with container locally

```sh
# Build container
docker build -t container-testing .

# Start a shell on container to try things out
docker run -it container-testing bash
```

## Pushing container updates to DockerHub repository

```sh
docker login

# Create tags for current build of container
docker image tag container-testing <org>/smartsim-testing:latest
docker image tag container-testing <org>/smartsim-testing:<version>

# Push current build of container with all tags created
docker image push --all-tags <org>/smartsim-testing
```


