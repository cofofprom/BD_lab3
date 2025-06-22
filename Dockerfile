FROM python:3.9-slim
USER root

WORKDIR /app

RUN apt update && \
    apt upgrade -y && \
    apt install -y openjdk-17-jre
    
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/ ./src/
COPY ./entrypoint.sh .

CMD [ "bash", "./entrypoint.sh" ]