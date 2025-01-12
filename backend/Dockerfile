ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /app
# Copy both requirements files at once
COPY requirements*.txt ./
# Install all requirements in one layer
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements-core.txt,target=requirements-core.txt \
    --mount=type=bind,source=requirements-app.txt,target=requirements-app.txt \
    python -m pip install --user -r requirements-core.txt && \
    python -m pip install --user -r requirements-app.txt

FROM python:${PYTHON_VERSION}-slim AS final
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/torch_home/model_weights/vgg
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

WORKDIR /app
# Combine directory creation and user setup
RUN mkdir -p /app/torch_home/model_weights/vgg /app/output_imgs /home/appuser/.local && \
    adduser \
        --disabled-password \
        --gecos "" \
        --home "/home/appuser" \
        --shell "/sbin/nologin" \
        --no-create-home \
        appuser && \
    chown -R appuser:appuser /app
# Single copy of packages to the correct location
COPY --from=builder /root/.local /home/appuser/.local
# Copy source code and fix permissions in one layer
COPY . .
RUN chown -R appuser:appuser /app /home/appuser
USER appuser
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload