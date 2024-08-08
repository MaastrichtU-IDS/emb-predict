ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/micheldumontier/emb-predict"
LABEL version="1.0"
LABEL description="Embedding-based prediction service"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam \
    PYTHONUNBUFFERED=1 \
    JOBLIB_TEMP_FOLDER=/app/data/tmp

# Change the current user to root and the working directory to /app
WORKDIR /app

# CUDA image required to install python
RUN apt-get update && \
    apt-get install -y vim curl wget unzip git nano

# setup GPU specific packages
RUN pip3 install --upgrade pip && \
    pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ && \
    pip3 install cupy-cuda12x && \
    pip3 install fastembed-gpu

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ADD . .
RUN pip3 install -e .

ENV PYTHONPATH=/app/src/
ENV CONFIG_FILE="config.production.yml"

CMD [ "uvicorn", "src.emb_predict.api:app", "--host", "0.0.0.0", "--port", "8808", "--reload" ]
#CMD [ "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--workers", "4", "src.emb_predict.api:app" ]
# uvicorn src.emb_predict.api:app --host "0.0.0.0" --port 8808
#ENTRYPOINT ["/bin/bash"]
