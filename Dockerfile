FROM python:3.8

ADD cardio_prediction.py .

RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install sklearn

CMD [ "python", "./cardio_prediction.py" ]