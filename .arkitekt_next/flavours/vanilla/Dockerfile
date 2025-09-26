FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime

RUN apt-get update
RUN apt-get install -y git
RUN mkdir /app
RUN git clone https://github.com/facebookresearch/sam2.git /app/sam2
WORKDIR /app/sam2
RUN pip install -e .
WORKDIR /app
RUN pip install "arkitekt-next[all]>=0.26.0"

RUN mkdir /real
WORKDIR /real
COPY .arkitekt_next /real/.arkitekt_next
COPY app.py /real/app.py
