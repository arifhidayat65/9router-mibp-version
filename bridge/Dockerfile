FROM python:3.12-slim
WORKDIR /app
COPY bridge.py .
ENV QUEUE_DIR=/app/queue
RUN mkdir -p /app/queue/pending /app/queue/done
EXPOSE 8765
CMD ["python3", "bridge.py"]
