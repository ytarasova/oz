FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY oz/ ./oz/
COPY static/ ./static/
COPY specs/ ./specs/
COPY main.py ./

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "python", "main.py"]
