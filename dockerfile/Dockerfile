FROM nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN cd /usr/local/ && apt-get update && apt-get install -y wget cmake libbz2-dev hdf5-tools pkg-config git python-dev build-essential python3-pip gfortran libblas-dev liblapack-dev vim emacs libx11-dev \
&& git clone -b release https://gitlab.com/petsc/petsc.git/ \
&& wget https://boostorg.jfrog.io/artifactory/main/release/1.71.0/source/boost_1_71_0.tar.gz \
&& tar -xf boost_1_71_0.tar.gz && cd boost_1_71_0 \
&& ./bootstrap.sh --prefix=/usr/local && ./b2 --prefix=/usr/local install \
&& rm /usr/local/boost_1_71_0.tar.gz && rm -rf /usr/localboost_1_71_0 && rm /usr/local/include/boost/mpi.hpp && rm -r /usr/local/include/boost/mpi \
&& cd /usr/local/petsc/ && ./configure --prefix=/usr/local --with-debugging=0 --download-adolc --download-colpack --download-triangle --download-zlib=yes --download-openmpi=yes --download-szlib=yes --with-boost-include=/usr/local/include/boost --with-boost-lib=/usr/local/ --with-x --with-cuda --download-mumps --download-eigen --download-gsl --download-hypre --download-scalapack --download-metis --download-parmetis \
&& make PETSC_DIR=/usr/local/petsc PETSC_ARCH=arch-linux-c-opt all \
&& make PETSC_DIR=/usr/local/petsc PETSC_ARCH=arch-linux-c-opt install \
&& rm -rf /usr/local/petsc/ \
&& cd /usr/local && git clone --branch release https://BachelorDog@bitbucket.org/fenics-project/ffc.git \
&& cd ffc && pip install --prefix=/usr/local . && rm -rf /usr/local/ffc \ 
&& cd /usr/local && git clone --branch release https://BachelorDog@bitbucket.org/fenics-project/dolfin.git \
&& cd dolfin && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && make install \
&& rm -rf /usr/local/dolfin
