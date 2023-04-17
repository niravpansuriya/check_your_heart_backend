FROM python:3.8

WORKDIR /python-docker

COPY docker_req.txt requirements.txt
COPY . .

RUN pip3 install tensorflow
RUN pip install -r requirements.txt

CMD [ "python3","application.py"]