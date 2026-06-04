# Fine-tuning-GenAI-model
Fine-tuning a GenAI model for customer support using LoRA

# Fine-Tuning FLAN-T5 for Customer Support Chatbot

## Project Overview

This project focuses on fine-tuning the **FLAN-T5 Base** model using **LoRA (Low-Rank Adaptation)** to build an intelligent customer support chatbot capable of answering common customer service queries.

The chatbot handles tasks such as:

* Order tracking
* Order cancellation
* Refund requests
* Return policies
* Damaged item reporting
* Subscription management
* Password recovery
* Account deletion
* Payment-related issues
* Delivery delays
* Human agent escalation

The project also integrates semantic intent matching, sentiment analysis, response evaluation, and a Gradio-based interactive interface.

---

## Objectives

* Fine-tune a pre-trained FLAN-T5 model for customer support tasks.
* Reduce training cost using LoRA adapters.
* Improve intent recognition through semantic similarity matching.
* Evaluate chatbot performance against baseline model responses.
* Deploy an interactive chatbot interface using Gradio.

---

## Project Pipeline

### 1. Dataset Creation

A synthetic customer support dataset was created containing:

* User queries
* Intent labels
* Ground-truth policy responses

Supported intents include:

| Intent                | Description                |
| --------------------- | -------------------------- |
| track_order           | Track package status       |
| cancel_order          | Cancel an order            |
| get_refund            | Refund inquiries           |
| return_policy         | Product return information |
| damaged_item          | Report damaged products    |
| cancel_subscription   | Subscription cancellation  |
| recover_password      | Password reset support     |
| delete_account        | Account deletion requests  |
| contact_human         | Connect with support agent |
| payment_failed        | Failed payment assistance  |
| check_payment_methods | Available payment methods  |
| delivery_delay        | Delivery delay support     |
| greeting              | Welcome interaction        |
| bot_identity          | Chatbot introduction       |

---

### 2. Data Preparation

The dataset is processed by:

* Cleaning user queries
* Generating paraphrased customer requests
* Tokenizing inputs and outputs
* Creating train-test splits
* Saving datasets for future training

---

### 3. Model Fine-Tuning

Base model:

```text
google/flan-t5-base
```

Fine-tuning technique:

```text
LoRA (Low-Rank Adaptation)
```

Key configuration:

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| LoRA Rank (r)     | 16                        |
| LoRA Alpha        | 32                        |
| LoRA Dropout      | 0.05                      |
| Task Type         | Seq2Seq Language Modeling |
| Target Modules    | q, v, o                   |
| Max Input Length  | 128                       |
| Max Output Length | 96                        |

Benefits of LoRA:

* Significantly fewer trainable parameters
* Lower GPU memory consumption
* Faster training
* Easier deployment

---

### 4. Smart Intent Detection

The chatbot combines:

#### Semantic Similarity Matching

Using:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

This enables the chatbot to:

* Understand paraphrased customer queries
* Match user requests to known intents
* Handle unseen phrasing effectively

---

### 5. Sentiment Analysis

A sentiment classifier is incorporated using:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

Purpose:

* Detect customer frustration
* Enable more empathetic responses
* Improve user experience

---

### 6. Response Generation

Workflow:

```text
User Query
      ↓
Intent Detection
      ↓
Policy Retrieval
      ↓
FLAN-T5 LoRA Model
      ↓
Final Response
```

The system generates policy-consistent customer support responses while maintaining conversational quality.

---

## Evaluation

The project evaluates:

### Automatic Metrics

* BLEU Score
* ROUGE Score

### LLM-Based Evaluation

Responses are judged on:

* Helpfulness
* Accuracy
* Empathy
* Conciseness

The evaluation compares:

1. Base FLAN-T5 responses
2. Fine-tuned FLAN-T5 + LoRA responses

---

## Gradio Chat Interface

The chatbot includes a Gradio application for real-time interaction.

Features:

* Conversational interface
* Multi-turn chat
* Memory support
* Customer support simulation
* Easy deployment

---

## Technologies Used

| Category           | Tools                     |
| ------------------ | ------------------------- |
| Language Model     | FLAN-T5 Base              |
| Fine-Tuning        | PEFT, LoRA                |
| Deep Learning      | PyTorch                   |
| Transformers       | Hugging Face Transformers |
| Datasets           | Hugging Face Datasets     |
| Semantic Search    | Sentence Transformers     |
| Sentiment Analysis | DistilBERT                |
| Evaluation         | BLEU, ROUGE               |
| Interface          | Gradio                    |
| Environment        | Google Colab              |

---

## Project Structure

```text
customer_support_chatbot/
│
├── datasets/
│   └── clean_policy_500/
│
├── models/
│   ├── lora_adapter_v4/
│   └── intent_classifier/
│
├── logs/
│   └── training_logs/
│
├── evaluation/
│   └── evaluation_results.json
│
├── app/
│   └── gradio_demo.py
│
└── Fine_Tuning_GenAI.ipynb
```

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/yourusername/customer-support-chatbot.git
cd customer-support-chatbot
```

### Install Dependencies

```bash
pip install transformers
pip install datasets
pip install peft
pip install sentence-transformers
pip install evaluate
pip install gradio
pip install accelerate
```

### Run Training

Execute notebook cells sequentially:

```text
Cell A → Setup
Cell B → Dataset Creation
Cell C → LoRA Fine-Tuning
Cell D → Smart Pipeline
Fix Cell → Intent Improvement
Evaluation Cell → Performance Analysis
Cell E → Gradio Deployment
```

---

## Expected Outcomes

* Accurate intent classification
* Fast and memory-efficient fine-tuning
* Improved customer support response quality
* Better policy adherence than the base model
* Interactive chatbot deployment with Gradio

---

## Future Improvements

* Multi-language support
* Retrieval-Augmented Generation (RAG)
* Integration with real customer service databases
* Human feedback-based reinforcement learning
* Deployment on cloud platforms (AWS, GCP, Azure)

---




