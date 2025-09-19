FROM python:3.11-slim-buster

WORKDIR /app


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .


ENV DOMAIN=127.0.0.1
ENV PORT=5000
ENV PREFIX=""
ENV DB_HOST=localhost
ENV DB_PORT=3306
ENV DB_USER=root
ENV DB_PASSWORD=pass@word1
ENV DB_NAME=bookdb

EXPOSE 5000

CMD ["python3","-m","flask","run","--host=0.0.0.0"]

