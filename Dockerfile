FROM ubuntu:22.04

# non interactive installation (no ask for time zone)
ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

# install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    npm \
    libginac11 \
    wget \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl  \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install pyenv
RUN mkdir /pyenvsrc && \
    cd /pyenvsrc && \
    wget https://github.com/pyenv/pyenv/archive/refs/tags/v2.6.1.tar.gz && \
    tar xzvf v2.6.1.tar.gz && \
    mv ./pyenv-2.6.1 /.pyenv

#git clone git://github.com/yyuu/pyenv.git /.pyenv

ENV PYENV_ROOT=/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# install python 3.11
RUN pyenv install 3.11.13 && \
    pyenv global 3.11.13 && \
    pyenv rehash

# cp source folder
COPY . /zorro

# install node dependencies
RUN cd /zorro && \
    npm install

# create virtual environment and install python dependencies
RUN cd /zorro && \
    python -m venv venv && \
    . ./venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# application's webport
EXPOSE 3000

# set entry point
WORKDIR /zorro
ENTRYPOINT [ "/zorro/zorro.sh" ]
