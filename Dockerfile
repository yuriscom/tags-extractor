FROM amazonlinux:latest

ARG PYTHON_GLOBAL_VERSION=3.9
ARG PYTHON_VERSION=3.9.7

RUN yum -y update && \
    yum -y groupinstall "Development Tools" && \
    yum -y install openssl-devel bzip2-devel libffi-devel && \
    yum -y install wget && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xvf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_GLOBAL_VERSION}*/ && \
    ./configure --enable-optimizations && \
    en_core_web_smmake altinstall && \
    yum install -y zip && \
    yum clean all;


RUN python${PYTHON_GLOBAL_VERSION} -m pip install --upgrade pip && \
    python${PYTHON_GLOBAL_VERSION} -m pip install virtualenv
