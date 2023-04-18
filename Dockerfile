FROM python:3.9.6-slim

ADD . /chorus
WORKDIR /chorus

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install sudo ufw build-essential wget libpq-dev python3-pip python3-dev libpython3-dev graphviz -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /chorus/pam
