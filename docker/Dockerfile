FROM python:3.9

# Install system packages
RUN \
	apt-get update -y && \
	apt-get install -y libgl1 gcc build-essential logrotate && \
	apt-get install -y jq

COPY ./requirements /tmp/requirements
COPY docker/logrotate.conf /etc/logrotate.d/id-scanner

WORKDIR /app
ARG REQUIREMENTS_FILE
RUN \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir restricted_pkg && \
    pip install --no-cache-dir -r /tmp/requirements/$REQUIREMENTS_FILE

# Configure logrotate
RUN \
    mkdir -p /var/log/id-scanner/ && \
    chmod 0444 /etc/logrotate.d/id-scanner && \
    touch logrotate_task && \
    echo "* * * * * /usr/sbin/logrotate -f /etc/logrotate.conf 2>&1 | tee -a /var/log/cron.log" > logrotate_task && \
    crontab logrotate_task

COPY . /app
COPY ./docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

