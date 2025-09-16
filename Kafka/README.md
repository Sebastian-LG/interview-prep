# Apache Kafka Basics

## 📌 What is Apache Kafka?

Apache Kafka is a **distributed streaming platform** used to build real-time data pipelines and streaming applications. It is designed for high-throughput, fault-tolerant, and scalable messaging between systems.

Kafka is widely used for:

* **Real-time analytics**
* **Event-driven architectures**
* **Data streaming between microservices**
* **Log aggregation and monitoring**

---

## ⚙️ Core Concepts

### 1. **Producer**

* Application that **publishes (writes) data** into Kafka topics.

### 2. **Consumer**

* Application that **reads data** from Kafka topics.
* Consumers are usually grouped into **Consumer Groups** for load balancing.

### 3. **Topic**

* A logical channel where messages are published.
* Topics are divided into **Partitions** for parallelism and scalability.

### 4. **Partition**

* Each topic can have multiple partitions.
* Partitions ensure ordering of messages within them.

### 5. **Broker**

* A Kafka server that stores data and serves client requests.
* A Kafka cluster is made of multiple brokers.

### 6. **ZooKeeper / KRaft**

* Traditionally, **ZooKeeper** manages cluster metadata.
* Newer Kafka versions (2.8+) can run with **KRaft (Kafka Raft)** without ZooKeeper.

---

## 🚀 Kafka Workflow

1. **Producer** sends messages to a **Topic**.
2. Kafka stores the messages across **Partitions** in different **Brokers**.
3. **Consumers** read messages from partitions (in consumer groups).
4. Kafka ensures:

   * Scalability (via partitions)
   * Durability (via replication)
   * Fault tolerance

---

## 🔑 Key Features

* **High throughput** (millions of messages/sec)
* **Scalable** horizontally with more brokers/partitions
* **Durable storage** (data written to disk, replicated)
* **Fault-tolerant**
* **Real-time stream processing** (with Kafka Streams or ksqlDB)

---

## 🛠️ Basic Commands

### Start ZooKeeper (if not using KRaft)

```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

### Start Kafka Broker

```bash
bin/kafka-server-start.sh config/server.properties
```

### Create a Topic

```bash
bin/kafka-topics.sh --create --topic test-topic --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

### List Topics

```bash
bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

### Produce Messages

```bash
bin/kafka-console-producer.sh --topic test-topic --bootstrap-server localhost:9092
```

### Consume Messages

```bash
bin/kafka-console-consumer.sh --topic test-topic --from-beginning --bootstrap-server localhost:9092
```

---

## 📚 Learning Resources

* [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
* [Kafka Quickstart Guide](https://kafka.apache.org/quickstart)
* [Confluent Kafka Tutorials](https://developer.confluent.io/)

---

# ☁️ Kafka in the Cloud

## 1. **AWS (Amazon Web Services)**

### Option A: **Amazon MSK (Managed Streaming for Apache Kafka)**

* Fully managed Kafka service.
* AWS handles brokers, scaling, replication, and patching.
* You just create topics and connect apps.
* You can also run **Kafka Connect** inside MSK (via MSK Connect).

🔑 How to use:

1. Go to **AWS Console → MSK → Create Cluster**.
2. Choose number of brokers, partitions, etc.
3. Connect producers/consumers using the broker endpoint.
4. For integrations → use **MSK Connect** (deploy connectors easily).

👉 Example: MySQL on RDS → Kafka (MSK) → S3 using a JDBC Source + S3 Sink connector.

---

## 2. **Azure (Microsoft Cloud)**

### Option A: **Azure Event Hubs (Kafka API)**

* Not actual Kafka under the hood, but **Kafka clients work with it**.
* You connect to Event Hubs using Kafka libraries (producers/consumers).
* Great if your company is already deep into Azure.

🔑 How to use:

1. Create an **Event Hub Namespace** in the Azure portal.
2. Enable the **Kafka surface**.
3. Get connection strings (like bootstrap server + SAS token).
4. Use Kafka clients (Java, Python, etc.) to produce/consume.

### Option B: **Confluent Cloud on Azure**

* Confluent partners with Microsoft.
* If you need **Kafka Connect, ksqlDB, Schema Registry** → go with Confluent.

---

## 3. **GCP (Google Cloud Platform)**

### Option A: **Confluent Cloud on GCP**

* The most common way to run Kafka in GCP.
* Fully managed by Confluent.
* Comes with **Kafka Connect, ksqlDB, Schema Registry**.

### Option B: **GCP Pub/Sub (Kafka Alternative)**

* Native GCP service (not Kafka).
* Can integrate with Kafka using connectors.

🔑 How to use (Confluent Cloud):

1. Create a Confluent Cloud cluster in GCP.
2. Choose region, cluster type (Standard, Dedicated, etc.).
3. Use Kafka client libraries with the Confluent endpoint + API keys.
4. Add connectors (e.g., GCS Sink Connector to stream data into Google Cloud Storage).

---

# 📊 Summary Table

| Cloud     | Native Option              | Kafka API Compatible? | Best for…                                                      |
| --------- | -------------------------- | --------------------- | -------------------------------------------------------------- |
| **AWS**   | Amazon MSK (+ MSK Connect) | ✅ True Kafka          | Companies already using AWS infra                              |
| **Azure** | Event Hubs (Kafka API)     | ✅ Works like Kafka    | Azure-heavy environments                                       |
| **GCP**   | Confluent Cloud on GCP     | ✅ True Kafka          | Data teams needing full Kafka stack (Connect, Schema, Streams) |

---

## ✅ Key Takeaway

* **AWS MSK** → most common in cloud jobs (Kafka “as is”).
* **Azure Event Hubs** → looks like Kafka, good for Microsoft shops.
* **Confluent Cloud (multi-cloud: AWS, Azure, GCP)** → full Kafka ecosystem with connectors, ksqlDB, schema registry.


