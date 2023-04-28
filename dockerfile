FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
RUN apt-get update \
    && pip install gpustat \
    && apt-get -y install tmux \
    && apt-get -y install git \
    && apt-get -y install gcc \
    && apt-get -y install libsm6 \
    && apt-get -y install libxext-dev \
    && apt-get -y install libxrender1 \
    && apt-get -y install libglib2.0-dev \
    && apt-get -y install default-jre \
    && pip install transformers==4.27.0 \
    && pip install lmdb==1.4.0 \
    && pip install clip \
    && pip install ftfy==6.1.1 \
    && pip install albumentations==1.3.0 \
    && pip install classy-vision==0.6.0 \
    && pip install gpustat==1.0.0 \
    && conda install -y faiss-gpu -c pytorch \
    && pip install yapf==0.32.0 \
    && pip install pytorch-lightning==1.5.7 \
    && pip install pytorch-metric-learning==0.9.99 \
    && pip install torchmetrics==0.11.0 \
    && pip install tslearn==0.5.2 \
    && pip install tensorboard==2.11.2 \
    && pip install tensorboardx==2.5.1 \
    && pip install torchsnooper==0.8 \
    && pip install yacs==0.1.8 \
    && pip install setuptools==59.5.0 \
    && conda install -y pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch \
    && pip install -U openmim \
    && mim install mmcv==1.7.1 \
    && pip install timm==0.6.12 \
    && pip install augly==1.0.0 \
    && pip install matplotlib \
