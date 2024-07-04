FROM apache/airflow:2.9.1
ADD requirements.txt .
ADD taskBash.py /tmp/
RUN pip install apache-airflow==2.9.1 -r requirements.txt
# RUN cd tasks && pip install -e .
# RUN cd ~ && cd pipelines && pip install -e .