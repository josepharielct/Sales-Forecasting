FROM python:3.9-slim-buster

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r app/requirements.txt

############################

WORKDIR /app

COPY ./ /app/


EXPOSE 80

CMD ["uvicorn", "main_API:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
