FROM neurips23

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10 nlohmann-json3-dev libeigen3-dev

ADD https://api.github.com/repos/jacketsj/expANN/git/refs/heads/pybinds version.json
RUN git clone -b pybinds https://github.com/jacketsj/expANN.git
RUN cd expANN && git submodule update --init --recursive
RUN cd expANN && pip install pybind11
RUN cd expANN && bash build.sh

ENV PYTHONPATH=$PYTHONPATH:/home/app/expANN/build

WORKDIR /home/app
