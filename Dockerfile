# ---------------------------------------------------#
# Set the base image
# ---------------------------------------------------#
FROM python:3.7-stretch


# ---------------------------------------------------#
# Install data/models dependencies
# ---------------------------------------------------#
RUN mkdir -p /models
ADD models/all_tversky_9959.hdf5 /models/all_tversky_9959.hdf5
ADD models/head_ce_9963.hdf5 /models/head_ce_9963.hdf5
ADD models/op_ce_9989.hdf5 /models/op_ce_9989.hdf5
ADD models/head_tversky_9963.hdf5 /models/head_tversky_9963.hdf5


# ---------------------------------------------------#
# Install scripts and setup entry point
# ---------------------------------------------------#
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD utils.py /app/utils.py
ADD loss_functions.py /app/loss_functions.py
ADD run.py /app/run.py


# ---------------------------------------------------#
# Use pip to install DL/common dependencies
# ---------------------------------------------------#

RUN pip install numpy==1.17.4
RUN pip install pydicom==1.4.2
RUN pip install pillow==6.2.1
RUN pip install shapely==1.7.0
RUN pip install future==0.17.1
RUN pip install matplotlib==3.1.1
RUN pip install tensorflow==2.0.0


COPY ./requirements.txt /requirements.txt

# Install the project dependency listed in requirements.txt
# RUN pip install --ignore-installed --default-timeout=10000 -r /requirements.txt

RUN pip install --upgrade --force-reinstall --default-timeout=10000 -r /requirements.txt


# ---------------------------------------------------#
# Install Cytomine python client
# ---------------------------------------------------#
RUN git clone https://github.com/Cytomine-ULiege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.7.3 && pip install . && \
    rm -r /Cytomine-python-client



# ---------------------------------------------------#
# 7. Set entrypoint to ">python /app/run.py"
# ---------------------------------------------------#
ENTRYPOINT ["python", "/app/run.py"]