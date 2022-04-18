FROM nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN cd /usr/local/ && apt-get update && apt-get install -y cmake libbz2-dev libboost1.71-all-dev hdf5-tools pkg-config git python-dev build-essential python3-pip gfortran libblas-dev liblapack-dev vim emacs libx11-dev \
&& git clone -b release https://gitlab.com/petsc/petsc.git/ \
&& cd /usr/local/petsc/ && ./configure --prefix=/usr/local --with-mpi --with-debugging=0 --download-adolc --download-colpack --download-triangle --download-zlib=yes --download-szlib=yes --with-x --with-cuda --download-mumps --download-eigen --download-gsl --download-hypre --download-scalapack --download-metis --download-parmetis \
&& make PETSC_DIR=/usr/local/petsc PETSC_ARCH=arch-linux-c-opt all \
&& make PETSC_DIR=/usr/local/petsc PETSC_ARCH=arch-linux-c-opt install \
&& rm -rf /usr/local/petsc/ \
&& cd /usr/local && git clone --branch release https://BachelorDog@bitbucket.org/fenics-project/ffc.git \
&& cd ffc && pip install --prefix=/usr/local . && rm -rf /usr/local/ffc 
&& cd /usr/local && git clone --branch release https://BachelorDog@bitbucket.org/fenics-project/dolfin.git \
&& cd dolfin && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && make install \
&& rm -rf /usr/local/dolfin  && source /usr/local/share/dolfin/dolfin.conf
