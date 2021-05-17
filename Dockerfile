FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7
RUN pip install --upgrade pip

RUN pip install neuralpredictors
RUN pip install nnfabrik

ADD ./nnsysident /src/nnsysident
RUN pip install -e /src/nnsysident

ADD ./data_port /src/data_port
RUN pip install -e /src/data_port

RUN pip3 --no-cache-dir install statsmodels
RUN pip3 install ax_platform
RUN pip3 install hiplot
