# Makefile for Docker operations

.PHONY: build run stop clean all

# Build the Docker image
build:
	docker build -t llmprompt-nexus -f docker/Dockerfile .

# Run the container with host network
run:
	docker run --name llmprompt-nexus --network=host -v ./:/llmprompt-nexus -d llmprompt-nexus


# Run the container interactively
run-interactive:
	docker run --name llmprompt-nexus --network=host -v ./:/llmprompt-nexus -it llmprompt-nexus /bin/bash

# Stop the container
stop:
	docker stop llmprompt-nexus
	docker rm llmprompt-nexus

# Remove the image
clean: stop
	docker rmi llmprompt-nexus

# Build and run
all: build run-interactive 