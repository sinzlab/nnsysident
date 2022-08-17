FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7
RUN pip install --upgrade pip

ADD ./data_port /src/data_port
RUN pip install -e /src/data_port

#RUN pip install git+https://github.com/sinzlab/neuralpredictors.git@v0.3.0
RUN pip install neuralpredictors

RUN pip install nnfabrik==0.1.0
RUN pip install figrid

RUN pip3 --no-cache-dir install \
    statsmodels \
    ax_platform \
    hiplot

# install the current project
WORKDIR /project
RUN mkdir /project/nnsysident
COPY ./nnsysident/nnsysident /project/nnsysident
COPY ./nnsysident/setup.py /project
RUN python -m pip install -e /project