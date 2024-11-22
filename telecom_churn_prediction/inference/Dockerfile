FROM python:3.10.12

# RUN mkdir model
# COPY model model

RUN python3 -V

COPY code  ./
COPY requirements.txt ./

RUN pip3 install -r requirements.txt

EXPOSE 8080

ENTRYPOINT [ "python3", "api.py" ]