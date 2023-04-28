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
