FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    ffprobe \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version \
    && which ffmpeg \
    && which ffprobe

ENV PATH="/usr/bin:${PATH}"
ENV FFMPEG_BINARY="/usr/bin/ffmpeg"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p ./temp_audio

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
