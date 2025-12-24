FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate environment
ENV PATH /opt/conda/envs/quantum-aco-dr/bin:$PATH
RUN echo "source activate quantum-aco-dr" > ~/.bashrc

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Expose ports
EXPOSE 8501  # Streamlit
EXPOSE 8000  # FastAPI

# Default command
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
