FROM python:3.9
RUN apt-get update
#RUN apt-get update && rm -rf /var/lib/apt/listls/*
RUN mkdir /opt/catalog_v3/
COPY . /opt/catalog_v3/

RUN pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
RUN pip3 install -r /opt/catalog_v3/requirements.txt
#RUN pip3 --no-cache-dir install -r /opt/catalog_v3/requirements.txt

WORKDIR /opt/catalog_v3/