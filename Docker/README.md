# 🐳 Containers & Docker Guide

## 📦 Containers

* Simplify **dependency installation** and configuration, avoiding manual errors when moving apps across environments (Dev → QA → Prod).
* Package applications with all dependencies and configuration.
* Portable artifact → can be shared or moved to any environment.
* Makes **development and deployment** faster and more reliable.
* Concept: **You pack, another developer unpacks.**

---

## 🖥️ Virtual Machines vs Containers

* Both use **virtualization**.
* **VMs:** Virtualize at the **hardware + OS kernel** level. Each VM has its own OS.
* **Containers:** Virtualize only the **application layer** (share the host OS kernel).
* **Portability:** VMs run on any OS, while Docker may have compatibility issues (e.g., older Windows versions).
* **Performance:** Containers are lighter, start faster, and use fewer resources than VMs.

---

## 🏗️ Docker Images

* A Docker image is built in **layers** (e.g., base Linux layer + MongoDB layer).
* A collection of layers = **Docker image**.
* Images are portable artifacts that can be moved/shared.
* Running an image → starts the app → creates a **container**.
* **Advantages:**

  * Very small in size.
  * Start and run faster than VMs.

---

## ⚙️ Docker Setup

* Ensure **system requirements** are met (e.g., Hyper-V must be enabled).

---

## 🛠️ Basic Commands

### Pull an image

```bash
docker pull docker/getting-started:pwd
```

### Run an image

```bash
docker run -d -p 80:80 docker/getting-started
```

* `-d`: Detached mode (continue using the terminal).
* `-p`: Port mapping → `hostPort:containerPort` (local port must be unique).

### Check running containers

```bash
docker ps
```

### List & remove images

```bash
docker images
docker image rm IMAGE_ID
```

### Stop a container

```bash
docker stop CONTAINER_ID
```

---

## 🏗️ Build an Image

```bash
docker build -t dockerimagename .
docker run -p 5000:5000 dockerimagename
```

* Generates two IPs: one **local**, one **inside the container**.

---

## ☁️ Publish to Docker Hub

```bash
docker login
```

### Rename image

```bash
docker tag dockerimagename dockerimagenewname
```

### Push image

```bash
docker push dockerimagename:latest
```

(or specify a version number instead of `latest`)

---

## ⚡ Docker Compose

### Why Docker Compose?

* Manage **multi-container applications** (e.g., app + database + cache).
* Define services, networks, and volumes in a **single YAML file**.
* Makes running complex stacks **easy and reproducible**.

### Key Commands

```bash
docker compose up -d --build  # Start all services in detached mode
docker compose stop    # Stop all services
docker compose down    # Stop & remove containers, networks, volumes
```

### Example `docker-compose.yaml`

```yaml
version: "3.9"
services:
  web:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
```
