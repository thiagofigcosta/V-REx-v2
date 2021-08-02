FROM python:3.8 AS builder

COPY requirements.txt .

RUN pip install --user -r requirements.txt

FROM python:3.8-slim
WORKDIR /code

COPY --from=builder /root/.local /root/.local
COPY Pytho{N}.py .
COPY entrypoint.sh .
COPY frontstarter.sh .
COPY src/ .

RUN chmod +x entrypoint.sh
RUN chmod +x frontstarter.sh

RUN cp frontstarter.sh /usr/bin/front-end
RUN cp frontstarter.sh /usr/bin/frontend
RUN cp frontstarter.sh /usr/bin/front

ENV PATH=/root/.local/bin:$PATH

RUN cp /bin/bash /bin/bash_bkp
RUN chmod +x /bin/bash_bkp
RUN rm -f /bin/bash && cp frontstarter.sh /bin/bash

CMD ["./entrypoint.sh"]