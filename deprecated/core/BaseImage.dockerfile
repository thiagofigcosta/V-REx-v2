FROM ubuntu:20.04 AS ubuntu_2004_vrex_core_build
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get install -y --no-install-recommends gcc-9 g++-9 build-essential cmake libboost-stacktrace-dev libssl-dev libsasl2-dev python-dev libboost-filesystem-dev libboost-all-dev python3-pip python3-dev
RUN apt-get install -y --no-install-recommends gdb valgrind
RUN apt-get install -y libgomp1

RUN mkdir /cpp_libs
COPY src/mongo-c-driver /cpp_libs/mongo-c-driver
RUN mkdir -p /cpp_libs/mongo-c-driver/cmake-build 
RUN cd /cpp_libs/mongo-c-driver/cmake-build ; \
 cmake -D ENABLE_AUTOMATIC_INIT_AND_CLEANUP=OFF .. &&  \
 cmake --build . && \
 cmake --build . --target install

COPY src/mongo-cxx-driver /cpp_libs/mongo-cxx-driver
COPY src/mnmlstc_core /cpp_libs/mnmlstc_core
RUN mkdir -p /cpp_libs/mongo-cxx-driver/build
RUN cd /cpp_libs/mongo-cxx-driver/build ; \
 cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_CXX_COMPILER="/usr/bin/g++" .. && \
 cmake --build . --target EP_mnmlstc_core && \
 cmake --build . && \
 cmake --build . --target install

# RUN rm -f /usr/local/lib/libmongocxx.so._noabi ; ln -s /usr/local/lib/libmogocxx.so.3.6.2 /usr/local/lib/libmongocxx.so._noabi
# RUN rm -f /usr/local/lib/libmongocxx.so ; ln -s /usr/local/lib/libmongocxx.so._noabi /usr/local/lib/libmongocxx.so
# # RUN ls -R | awk '/:$/&&f{s=$0;f=0}/:$/&&!f{sub(/:$/,"");s=$0;f=1;next}NF&&f{ print s"/"$0 }' # show folder structure

CMD ["tail","-f","/dev/null"]