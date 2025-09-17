FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Copy project files into the container
COPY . /workspace

# Install dependencies from the pinned requirements file
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Default command: drop into a shell; override in docker run
CMD ["bash"]