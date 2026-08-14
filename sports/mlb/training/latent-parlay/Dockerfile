FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m pip install --upgrade pip \
    && python -m pip install pandas==2.3.1 scikit-learn==1.7.1

WORKDIR /workspace

ENTRYPOINT ["python", "sports/mlb/scripts/train_latent_parlay_model.py"]
