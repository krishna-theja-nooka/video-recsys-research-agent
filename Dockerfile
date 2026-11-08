FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY reclab ./reclab
RUN pip install --no-deps --no-build-isolation .
ENTRYPOINT ["python", "-m", "reclab.cli"]
CMD ["--help"]
