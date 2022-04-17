FROM nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN cd /usr/local/ && apt-get update && apt-get install -y cmake libbz2-dev git python-dev build-essential python3-pybind11 python3-pip gfortran libblas-dev liblapack-dev vim emacs libx11-dev \
&& git clone -b release https://gitlab.com/petsc/petsc.git/ \
&& cd /usr/local/petsc/ && ./configure --prefix=/usr/local --with-debugging=0 --download-adolc --download-colpack --download-triangle --download-zlib=yes --download-hdf5=yes --download-szlib=yes --with-x --with-cuda --download-openmpi --download-mumps --download-boost --download-eigen --download-gsl --download-hypre --download-scalapack --download-metis --download-parmetis \
&& make PETSC_DIR=/usr/local/petsc PETSC_ARCH=arch-linux-c-opt all \
&& make PETSC_DIR=/usr/local/petsc PETSC_ARCH=arch-linux-c-opt install \
&& rm -rf /usr/local/petsc/ \
&& cd /usr/local && git clone https://github.com/xtensor-stack/xtl.git && cd xtl \
&& cmake -DCMAKE_INSTALL_PREFIX=/usr/local && make install \
&& rm -rf /usr/local/xtl/ \
&& cd /usr/local && git clone https://github.com/xtensor-stack/xtensor.git && cd xtensor \
&& cmake -DCMAKE_INSTALL_PREFIX=/usr/local && make install \
&& rm -rf /usr/local/xtensor/ \
&& cd /usr/local && git clone https://github.com/FEniCS/basix.git && cd basix \
&& cmake -DCMAKE_INSTALL_PREFIX=/usr/local && make install \
&& rm -rf /usr/local/basix \
&& cd /usr/local && git clone https://github.com/FEniCS/ufl.git && cd ufl \
&& pip install . && rm -rf /usr/local/ufl \
&& cd /usr/local && git clone https://github.com/FEniCS/ffcx.git && cd ffcx/cmake \
&& cmake -DCMAKE_INSTALL_PREFIX=/usr/local && make install \
&& rm -rf /usr/local/ffcx \
&& cd /usr/local && git clone https://github.com/ornladios/ADIOS2.git ADIOS2 && cd ADIOS2 && mkdir build && cd build \
&& cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && make install \
&& rm -rf /usr/local/ADIOS2 \
&& cd /usr/local && git clone https://github.com/FEniCS/dolfinx.git && cd dolfinx/cpp && mkdir build && cd build \
&& cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && make install \
&& rm -rf /usr/local/dolfinx 

