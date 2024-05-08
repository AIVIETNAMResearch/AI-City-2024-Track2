FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

WORKDIR /home
COPY src/preprocess/segment_extraction/requirements.txt /home/

ENV PATH="${HOME}/miniconda3/bin:${PATH}"
ARG PATH="${HOME}/miniconda3/bin:${PATH}"

RUN apt-get update &&  \
    apt-get upgrade -y &&  \
    apt-get install -y \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    gcc \
    git \
    locales \
    net-tools \
    wget \
    libpq-dev \
    libsndfile1-dev \
    git \
    git-lfs \
    libgl1 \
    unzip \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && sh Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH /home/miniconda/bin:$PATH

RUN conda init bash && \
    . /root/.bashrc && \
    conda update conda -y && \
    conda create -n segment_extraction python=3.10.14 pip -y && \
    conda activate segment_extraction && \
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    pip install flash-attn --no-build-isolation

RUN rm requirements.txt
RUN echo 'conda activate segment_extraction' >> /root/.bashrc