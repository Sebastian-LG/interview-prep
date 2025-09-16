# MLflow

**MLflow** is an open-source platform for managing the complete machine learning lifecycle, including experiment tracking, reproducible runs, model packaging, and deployment.

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Basic Concepts](#basic-concepts)
* [Typical Workflow](#typical-workflow)
* [Common Commands](#common-commands)

## Introduction

MLflow allows you to:

* **Track experiments**: Log parameters, metrics, and artifacts.
* **Package models**: Save and share models in a standard format.
* **Deploy models**: Easily deploy models to production.
* **Reproduce runs**: Ensure experiments are reproducible.

It works with any ML library, language, or deployment tool, making it highly flexible for data science workflows.

## Installation

### Prerequisites

* Python 3.8+

### Install MLflow

```bash
pip install mlflow
```

Check installation:

```bash
mlflow --version
```

Optionally, install additional dependencies for specific ML frameworks:

```bash
pip install mlflow[extras]
```

## Basic Concepts

* **Run**: A single execution of a model training script, storing parameters, metrics, and artifacts.
* **Experiment**: A collection of runs, used to organize experiments.
* **Artifact**: Files or outputs (models, plots, datasets) produced by a run.
* **Tracking Server**: Optional server to log and visualize runs centrally.
* **Model Registry**: A central repository to manage and version ML models.

## Typical Workflow

1. **Start MLflow UI**

```bash
mlflow ui
```

By default, it runs at `http://localhost:5000`.

2. **Set up an experiment**

```python
import mlflow

mlflow.set_experiment("my_experiment")
```

3. **Log a run**

```python
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

4. **View results**

Open the MLflow UI in your browser to see logged runs, metrics, parameters, and artifacts.

5. **Save and load models**

```python
import mlflow.sklearn
mlflow.sklearn.log_model(model, "my_model")

# Load model
loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/my_model")
```

6. **Use Model Registry (optional)**

* Register a model
* Stage it as “Production”
* Track versions and deploy easily

## Common Commands

| Command                                   | Description                        |
| ----------------------------------------- | ---------------------------------- |
| `mlflow ui`                               | Launch the MLflow tracking UI      |
| `mlflow run <project>`                    | Run an MLflow project              |
| `mlflow experiments list`                 | List all experiments               |
| `mlflow run <project> -P <param>=<value>` | Run with parameters                |
| `mlflow models serve -m <model>`          | Serve a model locally via REST API |
| `mlflow models predict -m <model>`        | Make predictions using a model     |
