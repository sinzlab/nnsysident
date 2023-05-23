FROM sinzlab/pytorch:v3.9-torch1.13.1-cuda11.7.0-dj0.12.9
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade six

#ADD ./neuralmetrics /src/neuralmetrics
#RUN pip install -e /src/neuralmetrics

ADD ./mei /src/mei
RUN pip install -e /src/mei

ADD ./data_port /src/data_port
RUN pip install -e /src/data_port

RUN pip install git+https://github.com/kklurz/neuralpredictors.git@main
RUN pip install git+https://github.com/sinzlab/nnvision.git@inception_loops

RUN pip install nnfabrik==0.1.0
RUN pip install figrid

RUN pip3 --no-cache-dir install \
    statsmodels \
    ax_platform \
    hiplot \
    wandb

RUN pip install --upgrade scikit-image
RUN pip install --upgrade numpy==1.23.5
RUN pip install --upgrade datajoint==0.13.1

# install the current project
WORKDIR /project
RUN mkdir /project/nnsysident
COPY ./nnsysident/nnsysident /project/nnsysident
COPY ./nnsysident/setup.py /project
COPY ./nnsysident/pyproject.toml /project

RUN python -m pip install -e /project