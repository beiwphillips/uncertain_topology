FROM ubuntu:latest

RUN apt update
RUN apt install python3 python3-distutils curl git g++ python3-dev swig make -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install virtualenv
RUN git clone https://bitbucket.org/dmaljovec/uncertain_topology.git
ENV VIRTUAL_ENV=/opt/utpy
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN cd uncertain_topology && git pull && pip install -r requirements.txt && python setup.py install
