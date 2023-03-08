FROM frolvlad/alpine-python-machinelearning

RUN apk add \
  gcc \
  libc-dev \
  linux-headers \
  python3-dev \
  mariadb-connector-c-dev \
  mariadb-client \
  uwsgi-python3

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8888
