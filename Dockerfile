FROM python:3.6.10
MAINTAINER Jason Chua (jason@coldspot.wtf)

#Installing all dependencies

RUN apt update && apt upgrade

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /opt/app

RUN pip install -r requirements.txt 

RUN pip install opencv-contrib-python


CMD [ "python", "./Facial Recognition.py" ]


