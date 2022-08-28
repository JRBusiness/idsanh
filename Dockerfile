FROM python:3.9.7-buster
WORKDIR .
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
COPY requirements.txt requirements.txt
COPY .env .env
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
COPY . .
EXPOSE 5020
CMD ["run.py"]
ENTRYPOINT ["python3"]
