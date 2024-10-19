FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime

RUN apt-get update
RUN apt-get install -y git
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install "arkitekt-next[all]>=0.8.10"

RUN mkdir /app
WORKDIR /app
COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
