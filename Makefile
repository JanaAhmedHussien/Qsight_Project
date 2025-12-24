SHELL := /bin/bash

CONDA ?= conda
ENV_FILE ?= conda_env.yml
ENV_NAME ?= quantum-aco-dr
QML_ENV_FILE ?= model/qml_qiskit_env.yml
QML_ENV_NAME ?= qml-qiskit
PYTHON ?= python
TRAIN_ARGS ?=
DOCKER_IMAGE ?= qsight-app

.PHONY: help env update-env qml-env train streamlit docker-build docker-run lint clean

help:
	@echo "Available targets:"
	@echo "  make env           # Create the main conda environment"
	@echo "  make update-env    # Update main conda environment"
	@echo "  make qml-env       # Create the QML/Qiskit helper environment"
	@echo "  make train         # Run production training pipeline"
	@echo "  make streamlit     # Launch Streamlit app"
	@echo "  make docker-build  # Build Docker image"
	@echo "  make docker-run    # Run Docker container with GPU support"
	@echo "  make lint          # Run black & flake8 checks"
	@echo "  make clean         # Remove Python cache files"

env:
	$(CONDA) env create --file $(ENV_FILE) || echo "Environment may already exist. Use 'make update-env' to refresh."

update-env:
	$(CONDA) env update --file $(ENV_FILE) --prune

qml-env:
	$(CONDA) env create --file $(QML_ENV_FILE) || echo "QML environment may already exist."

train:
	$(CONDA) run -n $(ENV_NAME) $(PYTHON) -m model.production.run_training $(TRAIN_ARGS)

streamlit:
	$(CONDA) run -n $(ENV_NAME) streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run --rm --gpus all -p 8501:8501 -p 8000:8000 -v $(PWD)/trained_model:/app/trained_model $(DOCKER_IMAGE)

lint:
	$(CONDA) run -n $(ENV_NAME) black --check model streamlit_app.py
	$(CONDA) run -n $(ENV_NAME) flake8 model

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
