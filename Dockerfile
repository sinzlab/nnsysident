FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools




ADD ./data_port /src/data_port
RUN pip install -e /src/data_port

RUN pip install git+https://github.com/sinzlab/neuralpredictors.git@main

RUN pip install nnfabrik==0.1.0
RUN pip install figrid

RUN pip3 --no-cache-dir install \
    statsmodels \
    ax_platform \
    hiplot \
    wandb

RUN pip install --upgrade scikit-image

# install the current project
WORKDIR /project
RUN mkdir /project/nnsysident
COPY ./nnsysident/nnsysident /project/nnsysident
COPY ./nnsysident/setup.py /project
COPY ./nnsysident/pyproject.toml /project

RUN python -m pip install -e /project