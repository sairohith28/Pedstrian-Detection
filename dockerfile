# System
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Installing Dependancies
RUN apt update
RUN apt-get clean

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y git wget python3-dev python3-pip unzip

RUN python3 -m pip install pip --upgrade
RUN pip3 install scikit-build
RUN pip3 install numpy
RUN pip install tensorflow==1.13.2
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install pillow
RUN pip3 install opencv-python-headless
RUN pip3 install glob2
RUN pip3 install tensorflow-object-detection-api

RUN apt-get update
RUN apt install -y cmake curl

# Install dependancies of your application here
# RUN pip3 install <dependancy1>
# RUN apt install <dependancy2>
ENV PYTHONIOENCODING=utf8
RUN echo "$FOO $BAR $PYTHONIOENCODING"

# Copy neccessary file like inference program etc to docker image
COPY demo/ /model/
