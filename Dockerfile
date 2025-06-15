# Dockerfile

# --- Stage 1: Build dependencies ---
    FROM python:3.11-slim as builder
    WORKDIR /app
    
    # Install uv, our fast package installer
    RUN pip install uv
    
    # Copy only the dependency definition file and install dependencies
    # This layer is cached and only re-runs if pyproject.toml changes
    COPY pyproject.toml .
    RUN uv pip install --system -e .
    
    # --- Stage 2: Final application image ---
    FROM python:3.11-slim
    WORKDIR /app
    
    # Copy the installed dependencies from the builder stage
    COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    COPY --from=builder /usr/local/bin /usr/local/bin
    
    # Copy the application source code
    COPY ./src ./src
    COPY ./config ./config
    COPY app.py .
    
    # Expose the Streamlit port
    EXPOSE 8501
    
    # Set the healthcheck for the Streamlit app
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
      CMD curl --fail http://localhost:8501/_stcore/health
    
    # Command to run the Streamlit application
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]