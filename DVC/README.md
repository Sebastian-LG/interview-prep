# DVC (Data Version Control)

**DVC** is a tool for versioning datasets, machine learning models, and pipelines. It works alongside Git to manage data efficiently without storing large files directly in the repository.

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Basic Concepts](#basic-concepts)
* [Typical Workflow](#typical-workflow)
* [Common Commands](#common-commands)

## Introduction

DVC allows you to:

* Track large datasets and machine learning models.
* Create reproducible pipelines.
* Store data in remote storage (S3, GCS, Azure, SSH, etc.).
* Keep your Git repository lightweight.

It is particularly useful for data science and machine learning projects where datasets change frequently.

## Installation

### Prerequisites

* Python 3.8+
* Git

### Install DVC

```bash
pip install dvc
```

Or system-wide:

```bash
sudo apt install dvc
```

Check installation:

```bash
dvc --version
```

## Basic Concepts

* **DVC-tracked files**: Data files and models tracked by DVC instead of Git.
* **`.dvc` files**: Small files that store metadata for each tracked file.
* **Remote storage**: Cloud or network storage where actual data is stored.
* **Pipeline**: A sequence of stages that define how data is processed.

## Typical Workflow

1. **Initialize DVC**

```bash
dvc init
```

2. **Track data**

```bash
dvc add path/to/data.csv
```

3. **Commit DVC metadata**

```bash
git add data.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
```

4. **Configure remote storage**

```bash
dvc remote add -d myremote s3://mybucket/path
```

5. **Push data to remote**

```bash
dvc push
```

6. **Pull data from remote**

```bash
dvc pull
```

7. **Create reproducible pipelines**

```bash
dvc run -n stage_name \
        -d script.py \
        -d data/input.csv \
        -o data/output.csv \
        python script.py
```

8. **Reproduce pipeline**

```bash
dvc repro
```

## Common Commands

| Command             | Description                               |
| ------------------- | ----------------------------------------- |
| `dvc init`          | Initialize DVC in a repository            |
| `dvc add <file>`    | Track a file with DVC                     |
| `dvc remove <file>` | Stop tracking a file                      |
| `dvc status`        | Check if data or pipelines have changed   |
| `dvc repro`         | Reproduce pipeline stages                 |
| `dvc push`          | Upload tracked data to remote storage     |
| `dvc pull`          | Download tracked data from remote storage |
| `dvc metrics show`  | Show metrics tracked by DVC               |
