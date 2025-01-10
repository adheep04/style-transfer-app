# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

# Install PyTorch dependencies first (slow to install but rarely change)
COPY requirements-core.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements-core.txt,target=requirements-core.txt \
    python -m pip install --user -r requirements-core.txt

# Install other dependencies (faster to install, change more often)
COPY requirements-app.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements-app.txt,target=requirements-app.txt \
    python -m pip install --user -r requirements-app.txt

FROM python:${PYTHON_VERSION}-slim AS final

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1
# set torch's home directory to this directory so it caches
# model weights here
ENV TORCH_HOME=/app/torch_home/model_weights/vgg

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN mkdir -p /app/torch_home/model_weights/vgg /app/output_imgs && \
    adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser && \
    # give user permission to edit this directory
    chown -R appuser:appuser /app/torch_home/model_weights/vgg && \
    chown -R appuser:appuser /app/output_imgs

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy the source code into the container.
COPY . .

# Make sure appuser owns all copied files
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
