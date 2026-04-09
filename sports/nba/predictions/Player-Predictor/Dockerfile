# syntax=docker/dockerfile:1.7
#
# Unified Dockerfile for Player-Predictor.
# Build base image only:
#   docker build --target base -t qdi-rust-catboost .
# Build app image:
#   docker build -t player-predictor .
# Optional custom parent:
#   docker build --build-arg BASE_IMAGE=<custom-base> -t player-predictor .

ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m pip install --upgrade pip && \
    python -m pip install catboost

WORKDIR /workspace

FROM base AS app
WORKDIR /workspace
COPY . /workspace/

