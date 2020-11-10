FROM ubuntu:18.04
# FROM continuumio/miniconda3
# FROM mcr.microsoft.com/azureml/o16n-base/python-slim:latest.japaneast

# Options for common setup script
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gcc make python3-dev

RUN which conda && conda update -n base -c defaults conda
COPY conda-dev.yml ./
RUN pip uninstall typing; conda env update -f conda-dev.yml -n base
COPY conda.yml ./
RUN conda env update -f conda.yml -n base
