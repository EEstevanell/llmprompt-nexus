# Makefile for Docker operations
IMAGE_NAME = llms-api-framework
CONTAINER_NAME = llms-api-framework

.PHONY: build run stop clean all

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

# Run the container with host network
run:
	docker run --name $(CONTAINER_NAME) --network=host -d $(IMAGE_NAME)

# Run the container interactively
run-interactive:
	docker run --name $(CONTAINER_NAME) --network=host -it $(IMAGE_NAME) /bin/bash

# Stop the container
stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

# Remove the image
clean: stop
	docker rmi $(IMAGE_NAME)

# Build and run
all: build run-interactive 