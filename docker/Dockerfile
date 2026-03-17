FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Build dependencies for TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN cd /tmp \
    && wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd /tmp && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /work

# Install Python dependencies (no editable install in prod)
COPY pyproject.toml .
RUN pip install --no-cache-dir ta-lib && pip install --no-cache-dir -e .

COPY src/ ./src/
COPY bot/ ./bot/
COPY training/ ./training/
COPY back_test/ ./back_test/

ENV PYTHONPATH=/work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["python", "-m", "bot.src.main"]
