FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Build dependencies for TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential autoconf automake libtool \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
# config.guess/config.sub を更新して ARM64 (M2 Lima 含む) に対応
RUN cd /tmp \
    && wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && wget -q -O config.guess 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' \
    && wget -q -O config.sub   'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' \
    && ./configure --prefix=/usr \
    && make && make install \
    && cd /tmp && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /work

# Install Python dependencies
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
