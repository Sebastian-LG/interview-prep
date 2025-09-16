# DagsHub

**DagsHub** is a platform for version control and collaboration in data science and machine learning projects. It combines **Git**, **DVC**, and **MLflow** into a unified environment for managing code, data, and experiments.

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Basic Concepts](#basic-concepts)
* [Typical Workflow](#typical-workflow)
* [Common Commands](#common-commands)

## Introduction

DagsHub allows you to:

* Track **datasets** and **ML models** with DVC.
* Track **experiments** and **metrics** with MLflow.
* Collaborate with Git for code and project management.
* Visualize data, experiments, and metrics directly in a web interface.

It is ideal for teams or individuals who want a central platform to manage all aspects of a data science project.

## Installation

### Prerequisites

* Python 3.8+
* Git
* DVC
* MLflow

### Install DagsHub CLI

```bash
pip install dagshub
```

Check installation:

```bash
dagshub --version
```

### Sign up

* Create an account at [https://dagshub.com](https://dagshub.com)
* Authenticate via CLI:

```bash
dagshub login
```

## Basic Concepts

* **Repository**: Git-based project hosted on DagsHub.
* **Dataset**: Large files or datasets tracked with DVC.
* **Experiment**: ML experiment tracked with MLflow.
* **Remote storage**: Cloud storage linked via DVC for storing large datasets and models.
* **Issues & Collaboration**: Integrated GitHub-like issue tracking and pull requests.

## Typical Workflow

1. **Create a repository on DagsHub**

```bash
dagshub repo create my-project
```

2. **Clone repository locally**

```bash
git clone https://dagshub.com/<username>/my-project.git
cd my-project
```

3. **Initialize DVC and MLflow (optional)**

```bash
dvc init
mlflow ui
```

4. **Track data with DVC**

```bash
dvc add data/my_dataset.csv
git add data/.gitignore my_dataset.csv.dvc
git commit -m "Track dataset with DVC"
dvc push
```

5. **Track experiments with MLflow**

```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

6. **Push code and updates to DagsHub**

```bash
git add .
git commit -m "Update project"
git push origin main
```

7. **Collaborate and visualize**

* View datasets, metrics, and experiments in the DagsHub web interface.
* Share your repository with team members for collaboration.

## Common Commands

| Command                      | Description                                |
| ---------------------------- | ------------------------------------------ |
| `dagshub login`              | Authenticate your CLI with DagsHub         |
| `dagshub repo create <name>` | Create a new repository                    |
| `dagshub repo clone <repo>`  | Clone a DagsHub repository                 |
| `dvc add <file>`             | Track a file with DVC                      |
| `dvc push`                   | Upload DVC-tracked files to remote storage |
| `mlflow ui`                  | Launch MLflow tracking UI                  |
| `git push`                   | Push code to DagsHub repository            |

