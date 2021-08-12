FROM ubuntu_2004_vrex_core_build

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

RUN mkdir /vrex
COPY res/iris.data /vrex/res/iris.data
COPY src/vrex /vrex/src/

RUN cd /vrex/src/ ; make compileAndLink
RUN chmod +x /vrex/src/bin/vrex

COPY src/core-controller /vrex/ctrl/

RUN pip3 install --user -r /vrex/ctrl/requirements.txt

COPY raw-core-entrypoint.sh /standalone-core-entrypoint.sh
RUN chmod +x /standalone-core-entrypoint.sh
RUN ln -s /standalone-core-entrypoint.sh /usr/bin/args
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV PATH=/root/.local/bin:$PATH
CMD ["./entrypoint.sh"]