FROM ubuntu:18.04

RUN apt update
RUN apt upgrade -y
RUN apt-get install -y \
  build-essential make vim g++ sudo libomp-dev cmake libopenmpi-dev \
  openmpi-common openmpi-bin libopenmpi-dev openssh-client openssh-server net-tools netcat iptables

VOLUME /home
WORKDIR /home
ENTRYPOINT tail -f /dev/null
