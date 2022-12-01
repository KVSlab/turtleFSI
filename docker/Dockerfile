FROM condaforge/mambaforge 


WORKDIR /tmp/

ENV DEBIAN_FRONTEND=noninteractive

# Install ssh (missing dependency to run conda envs)
RUN apt-get update && \
    apt-get install -y ssh build-essential

# Upgrade mamba
RUN mamba upgrade -y mamba

# Copy environment and requirements files into docker env
COPY . turtleFSI

# Update environment file with new environment name
RUN mamba env update --file ./turtleFSI/environment.yml --name dockerenv
SHELL ["mamba", "run", "-n", "dockerenv", "/bin/bash", "-c"]

RUN python3 -m pip install ./turtleFSI pytest

# Test turtleFSI
RUN python3 -m pytest ./turtleFSI/tests

RUN echo "source activate dockerenv" > ~/.bashrc