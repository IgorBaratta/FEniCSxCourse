#!/bin/bash

# make module visible to user
# pkgconf@1.8.0%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load pkgconf-1.8.0-gcc-12.2.0-r6oblgz
# ncurses@6.3%gcc@12.2.0~symlinks+termlib abi=none build_system=autotools arch=linux-centos7-ivybridge
module load ncurses-6.3-gcc-12.2.0-rpsk3qf
# ca-certificates-mozilla@2022-10-11%gcc@12.2.0 build_system=generic arch=linux-centos7-ivybridge
module load ca-certificates-mozilla-2022-10-11-gcc-12.2.0-cmymec7
# berkeley-db@18.1.40%gcc@12.2.0+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-centos7-ivybridge
module load berkeley-db-18.1.40-gcc-12.2.0-pt426z7
# libiconv@1.16%gcc@12.2.0 build_system=autotools libs=shared,static arch=linux-centos7-ivybridge
module load libiconv-1.16-gcc-12.2.0-q3hwmrg
# diffutils@3.8%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load diffutils-3.8-gcc-12.2.0-pxo34ea
# bzip2@1.0.8%gcc@12.2.0~debug~pic+shared build_system=generic arch=linux-centos7-ivybridge
module load bzip2-1.0.8-gcc-12.2.0-il3l5em
# readline@8.1.2%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load readline-8.1.2-gcc-12.2.0-yqvtl6d
# gdbm@1.23%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load gdbm-1.23-gcc-12.2.0-ifwajuc
# zlib@1.2.13%gcc@12.2.0+optimize+pic+shared build_system=makefile arch=linux-centos7-ivybridge
module load zlib-1.2.13-gcc-12.2.0-u7kjh4e
# perl@5.36.0%gcc@12.2.0+cpanm+shared+threads build_system=generic arch=linux-centos7-ivybridge
module load perl-5.36.0-gcc-12.2.0-s6exf4r
# openssl@1.1.1s%gcc@12.2.0~docs~shared build_system=generic certs=mozilla arch=linux-centos7-ivybridge
module load openssl-1.1.1s-gcc-12.2.0-r4kpfmo
# cmake@3.25.1%gcc@12.2.0~doc+ncurses+ownlibs~qt build_system=generic build_type=Release arch=linux-centos7-ivybridge
module load cmake-3.25.1-gcc-12.2.0-gd2obex
# cmake@3.25.0%gcc@12.2.0~doc+ncurses+ownlibs~qt build_system=generic build_type=Release arch=linux-centos7-ivybridge
module load cmake-3.25.0-gcc-12.2.0-ytweddi
# openblas@0.3.21%gcc@12.2.0~bignuma~consistent_fpcsr+fortran~ilp64+locking+pic+shared build_system=makefile patches=d3d9b15 symbol_suffix=none threads=none arch=linux-centos7-ivybridge
module load openblas-0.3.21-gcc-12.2.0-txw66pt
# fenics-basix@main%gcc@12.2.0~ipo build_system=cmake build_type=RelWithDebInfo arch=linux-centos7-ivybridge
module load fenics-basix-main-gcc-12.2.0-4pv4agf
# boost@1.80.0%gcc@12.2.0~atomic~chrono~clanglibcpp~container~context~contract~coroutine~date_time~debug~exception~fiber+filesystem~graph~graph_parallel~icu~iostreams~json~locale~log~math~mpi+multithreaded~nowide~numpy~pic+program_options~python~random~regex~serialization+shared~signals~singlethreaded~stacktrace~system~taggedlayout~test~thread+timer~type_erasure~versionedlayout~wave build_system=generic cxxstd=98 patches=a440f96 visibility=hidden arch=linux-centos7-ivybridge
module load boost-1.80.0-gcc-12.2.0-wox5fe7
# fenics-ufcx@main%gcc@12.2.0~ipo build_system=cmake build_type=RelWithDebInfo arch=linux-centos7-ivybridge
module load fenics-ufcx-main-gcc-12.2.0-6wp3mxa
# cpio@2.13%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load cpio-2.13-gcc-12.2.0-hjdxdm5
# intel-mpi@2019.10.317%gcc@12.2.0~external-libfabric build_system=generic arch=linux-centos7-ivybridge
module load intel-mpi-2019.10.317-gcc-12.2.0-hkevula
# hdf5@1.12.2%gcc@12.2.0~cxx~fortran~hl~ipo~java+mpi+shared~szip~threadsafe+tools api=default build_system=cmake build_type=RelWithDebInfo arch=linux-centos7-ivybridge
module load hdf5-1.12.2-gcc-12.2.0-xcb4knj
# ncurses@6.4%gcc@12.2.0~symlinks+termlib abi=none build_system=autotools arch=linux-centos7-ivybridge
module load ncurses-6.4-gcc-12.2.0-2qnwagm
# libiconv@1.17%gcc@12.2.0 build_system=autotools libs=shared,static arch=linux-centos7-ivybridge
module load libiconv-1.17-gcc-12.2.0-3wqatjk
# diffutils@3.8%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load diffutils-3.8-gcc-12.2.0-l2hwwac
# bzip2@1.0.8%gcc@12.2.0~debug~pic+shared build_system=generic arch=linux-centos7-ivybridge
module load bzip2-1.0.8-gcc-12.2.0-iuerx6x
# readline@8.2%gcc@12.2.0 build_system=autotools patches=bbf97f1 arch=linux-centos7-ivybridge
module load readline-8.2-gcc-12.2.0-uxbbhvt
# gdbm@1.23%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load gdbm-1.23-gcc-12.2.0-axfuzom
# perl@5.36.0%gcc@12.2.0+cpanm+open+shared+threads build_system=generic arch=linux-centos7-ivybridge
module load perl-5.36.0-gcc-12.2.0-qxjwymq
# openssl@1.1.1s%gcc@12.2.0~docs~shared build_system=generic certs=mozilla arch=linux-centos7-ivybridge
module load openssl-1.1.1s-gcc-12.2.0-rhymar3
# cmake@3.25.1%gcc@12.2.0~doc+ncurses+ownlibs~qt build_system=generic build_type=Release arch=linux-centos7-ivybridge
module load cmake-3.25.1-gcc-12.2.0-duodk5g
# metis@5.1.0%gcc@12.2.0~gdb~int64~ipo~real64+shared build_system=cmake build_type=RelWithDebInfo patches=4991da9,93a7903,b1225da arch=linux-centos7-ivybridge
module load metis-5.1.0-gcc-12.2.0-nk5hibg
# parmetis@4.0.3%gcc@12.2.0~gdb~int64~ipo+shared build_system=cmake build_type=RelWithDebInfo patches=4f89253,50ed208,704b84f arch=linux-centos7-ivybridge
module load parmetis-4.0.3-gcc-12.2.0-sfknekz
# hypre@2.27.0%gcc@12.2.0~complex~cuda~debug~fortran~gptune~int64~internal-superlu~mixedint+mpi~openmp~rocm+shared~superlu-dist~sycl~umpire~unified-memory build_system=autotools arch=linux-centos7-ivybridge
module load hypre-2.27.0-gcc-12.2.0-fivr6dj
# libmd@1.0.4%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load libmd-1.0.4-gcc-12.2.0-2femkdy
# libbsd@0.11.5%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load libbsd-0.11.5-gcc-12.2.0-hku6h5m
# expat@2.5.0%gcc@12.2.0+libbsd build_system=autotools arch=linux-centos7-ivybridge
module load expat-2.5.0-gcc-12.2.0-jinadu4
# xz@5.2.7%gcc@12.2.0~pic build_system=autotools libs=shared,static arch=linux-centos7-ivybridge
module load xz-5.2.7-gcc-12.2.0-si5ysnr
# libxml2@2.10.3%gcc@12.2.0~python build_system=autotools arch=linux-centos7-ivybridge
module load libxml2-2.10.3-gcc-12.2.0-cx4flod
# pigz@2.7%gcc@12.2.0 build_system=makefile arch=linux-centos7-ivybridge
module load pigz-2.7-gcc-12.2.0-c3ixfw2
# zstd@1.5.2%gcc@12.2.0+programs build_system=makefile compression=none libs=shared,static arch=linux-centos7-ivybridge
module load zstd-1.5.2-gcc-12.2.0-a6vxsh5
# tar@1.34%gcc@12.2.0 build_system=autotools zip=pigz arch=linux-centos7-ivybridge
module load tar-1.34-gcc-12.2.0-aagu52g
# gettext@0.21.1%gcc@12.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz build_system=autotools arch=linux-centos7-ivybridge
module load gettext-0.21.1-gcc-12.2.0-cm2o636
# libffi@3.4.3%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load libffi-3.4.3-gcc-12.2.0-vhyufgj
# libxcrypt@4.4.33%gcc@12.2.0~obsolete_api build_system=autotools arch=linux-centos7-ivybridge
module load libxcrypt-4.4.33-gcc-12.2.0-avqvbg4
# sqlite@3.40.0%gcc@12.2.0+column_metadata+dynamic_extensions+fts~functions+rtree build_system=autotools arch=linux-centos7-ivybridge
module load sqlite-3.40.0-gcc-12.2.0-53fphsv
# util-linux-uuid@2.38.1%gcc@12.2.0 build_system=autotools arch=linux-centos7-ivybridge
module load util-linux-uuid-2.38.1-gcc-12.2.0-gdbxwwi
# python@3.10.8%gcc@12.2.0+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~nis~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic patches=0d98e93,7d40923,f2fd060 arch=linux-centos7-ivybridge
module load python-3.10.8-gcc-12.2.0-aysx3jq
# petsc@3.18.2%gcc@12.2.0~X~batch~cgns~complex~cuda~debug+double~exodusii~fftw~fortran~giflib+hdf5~hpddm~hwloc+hypre~int64~jpeg~knl~kokkos~libpng~libyaml~memkind+metis~mkl-pardiso~mmg~moab~mpfr+mpi~mumps~openmp~p4est~parmmg~ptscotch~random123~rocm~saws+shared~strumpack~suite-sparse~tetgen~trilinos~valgrind build_system=generic clanguage=C arch=linux-centos7-ivybridge
module load petsc-3.18.2-gcc-12.2.0-2ixtwez
# pugixml@1.11.4%gcc@12.2.0~ipo+pic+shared build_system=cmake build_type=RelWithDebInfo arch=linux-centos7-ivybridge
module load pugixml-1.11.4-gcc-12.2.0-4etxgqs
# fenics-dolfinx@main%gcc@12.2.0~adios2~ipo~slepc build_system=cmake build_type=RelWithDebInfo partitioners=parmetis arch=linux-centos7-ivybridge
module load fenics-dolfinx-main-gcc-12.2.0-cv2wm7n
# py-pip@22.2.2%gcc@12.2.0 build_system=generic arch=linux-centos7-ivybridge
module load py-pip-22.2.2-gcc-12.2.0-ldxsyev
# py-setuptools@63.0.0%gcc@12.2.0 build_system=generic arch=linux-centos7-ivybridge
module load py-setuptools-63.0.0-gcc-12.2.0-crhpacx
# py-wheel@0.37.1%gcc@12.2.0 build_system=generic arch=linux-centos7-ivybridge
module load py-wheel-0.37.1-gcc-12.2.0-m5kmvdk
# py-pycparser@2.21%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-pycparser-2.21-gcc-12.2.0-ikbml6c
# py-cffi@1.15.1%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-cffi-1.15.1-gcc-12.2.0-hvla6u3
# py-cython@0.29.32%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-cython-0.29.32-gcc-12.2.0-lbmln2y
# py-numpy@1.24.1%gcc@12.2.0+blas+lapack build_system=python_pip patches=873745d arch=linux-centos7-ivybridge
module load py-numpy-1.24.1-gcc-12.2.0-kfxy4x7
# libxml2@2.10.3%gcc@12.2.0~python build_system=autotools arch=linux-centos7-ivybridge
module load libxml2-2.10.3-gcc-12.2.0-euikbud
# tar@1.34%gcc@12.2.0 build_system=autotools zip=pigz arch=linux-centos7-ivybridge
module load tar-1.34-gcc-12.2.0-phwjhv6
# gettext@0.21.1%gcc@12.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz build_system=autotools arch=linux-centos7-ivybridge
module load gettext-0.21.1-gcc-12.2.0-mxfeajh
# libxcrypt@4.4.33%gcc@12.2.0~obsolete_api build_system=autotools arch=linux-centos7-ivybridge
module load libxcrypt-4.4.33-gcc-12.2.0-dxw2tsl
# sqlite@3.40.0%gcc@12.2.0+column_metadata+dynamic_extensions+fts~functions+rtree build_system=autotools arch=linux-centos7-ivybridge
module load sqlite-3.40.0-gcc-12.2.0-refyjuo
# python@3.10.8%gcc@12.2.0+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~nis~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic patches=0d98e93,7d40923,f2fd060 arch=linux-centos7-ivybridge
module load python-3.10.8-gcc-12.2.0-biuz4ql
# re2c@2.2%gcc@12.2.0 build_system=generic arch=linux-centos7-ivybridge
module load re2c-2.2-gcc-12.2.0-tnimm53
# ninja@1.11.1%gcc@12.2.0+re2c build_system=generic arch=linux-centos7-ivybridge
module load ninja-1.11.1-gcc-12.2.0-ljsgm5g
# py-pybind11@2.10.1%gcc@12.2.0~ipo build_system=cmake build_type=RelWithDebInfo arch=linux-centos7-ivybridge
module load py-pybind11-2.10.1-gcc-12.2.0-24ke7tz
# py-fenics-basix@main%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-fenics-basix-main-gcc-12.2.0-lvnvd2x
# py-fenics-ufl@main%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-fenics-ufl-main-gcc-12.2.0-b23g5qv
# py-fenics-ffcx@main%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-fenics-ffcx-main-gcc-12.2.0-tx5c7km
# py-mpi4py@3.1.4%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-mpi4py-3.1.4-gcc-12.2.0-dwjrfpy
# py-petsc4py@3.18.2%gcc@12.2.0+mpi build_system=python_pip patches=d344e0e arch=linux-centos7-ivybridge
module load py-petsc4py-3.18.2-gcc-12.2.0-ejqaogn
# py-fenics-dolfinx@main%gcc@12.2.0 build_system=python_pip arch=linux-centos7-ivybridge
module load py-fenics-dolfinx-main-gcc-12.2.0-tiyyy53
