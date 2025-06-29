# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose the Gradio or FastAPI port (if needed)
EXPOSE 7860

# Default command (change mode if needed)
CMD ["python", "-m", "src.main", "--mode", "cost"]
