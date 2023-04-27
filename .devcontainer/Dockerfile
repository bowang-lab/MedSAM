FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt -y update
RUN apt install -y git wget
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

# setup conda env
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
SHELL ["/bin/bash", "-i", "-c"]
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
RUN source $HOME/miniconda/bin/activate && conda init
RUN conda create -n medsam python=3.10 -y
RUN echo "conda activate medsam" >> ~/.bashrc
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


# COPY ./docker/mambarc /root/.mambarc
# COPY ./docker/bashrc /root/.bashrc

# SHELL ["/bin/bash", "-i", "-c"]
# RUN micromamba create -n paddle_testing python=3.9
# RUN echo "micromamba activate paddle_testing" >> /root/.bashrc
# RUN micromamba install paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge

# # patch paddle：paddle.fliud.tests 在源码中存在，但是没有打包进whl，会导致测试过程中import不到出错
# RUN cd /root/ && git clone https://github.com/PaddlePaddle/Paddle
# RUN cd /root/ && cp -r Paddle/python/paddle/fluid/tests/ /root/micromamba/envs/paddle_testing/lib/python3.9/site-packages/paddle/fluid/

# RUN cd /root/ && git clone https://github.com/PaddlePaddle/PaddleDetection.git && cd PaddleDetection && git reset --hard 6e3708e962538f431ad9ec38f891cb1791525e2c
# RUN cd /root/ && git clone https://github.com/PaddlePaddle/PaddleSeg.git && cd PaddleSeg && git reset --hard a178d72855053881db628a232b742b2234f1f1f4

# COPY ./requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt

# RUN echo "cd /root/framework_testing" >> /root/.bashrc
# RUN echo 'export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))' >> /root/.bashrc
