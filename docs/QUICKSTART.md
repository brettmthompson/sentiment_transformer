# Quick Start Guide

This guide will help you quickly test the sentiment analysis transformer locally.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Local Testing (Without KServe)](#local-testing-without-kserve)
- [Running Locally (Development Server)](#running-locally-development-server)
- [Deployment Workflow](#deployment-workflow)

## Overview

This transformer accepts requests from clients using the **V1 protocol** with custom JSON (`{"texts": [...]}`), while communicating with the predictor using the **V2 inference protocol**. This allows for a user-friendly API while maintaining compatibility with KServe's V2 protocol for model serving.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (modern Python package manager)
- (Optional) Docker or Podman for containerization

## Local Testing (Without KServe)

For testing the transformer logic locally without deploying to KServe:

### Step 1: Install uv and Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies and the project itself
uv sync
```

### Step 2: Test the Transformer

Create a simple test for the transformer:

```python
# test_transformer.py
from sentiment_transformer import SentimentTransformer
from kserve import InferResponse, InferOutput, PredictorConfig
import numpy as np

# Initialize transformer with predictor config
predictor_config = PredictorConfig(
    predictor_host="localhost:8080",
    predictor_protocol="v2"
)

transformer = SentimentTransformer(
    name="sentiment-analysis",
    tokenizer_name="optimum/distilbert-base-uncased-finetuned-sst-2-english",
    predictor_config=predictor_config,
    include_star_rating=True  # Optional: include 1-5 star ratings in response
)

# Test preprocessing
payload = {
    "texts": [
        "This is amazing!",
        "I hate this product"
    ]
}

infer_request = transformer.preprocess(payload)
print("Preprocessed request:")
print(f"Model name: {infer_request.model_name}")
print(f"Number of inputs: {len(infer_request.inputs)}")
for inp in infer_request.inputs:
    print(f"  - {inp.name}: shape {inp.shape}")

# Test postprocessing
# Simulate model output (logits for negative, positive)
mock_logits = [
    [-2.1, 3.8],  # Strong positive
    [2.5, -1.3]   # Strong negative
]

mock_output = InferOutput(
    name="predict",
    shape=[2, 2],
    datatype="FP32",
    data=mock_logits
)

mock_response = InferResponse(
    model_name="sentiment-analysis",
    outputs=[mock_output]
)

result = transformer.postprocess(mock_response)
print("\nPostprocessed result:")
import json
print(json.dumps(result, indent=2))
```

Run it:
```bash
uv run python test_transformer.py
```

Expected output:
- Preprocessed inputs with `input_ids` and `attention_mask`
- Postprocessed predictions with sentiment labels, confidence scores, and star ratings (1-5) if `include_star_rating=True`

## Running Locally (Development Server)

For running the actual transformer server locally (requires a running predictor with expected model):

```bash
uv run python -m sentiment_transformer.transformer \
  --model_name sentiment-analysis \
  --predictor_host localhost:8080 \
  --predictor_protocol v2 \
  --tokenizer_name optimum/distilbert-base-uncased-finetuned-sst-2-english \
  --include_star_rating  # Optional: include 1-5 star ratings in response
```

Then test with a request:

```bash
curl -X POST http://localhost:8080/v1/models/sentiment-analysis:predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This product is amazing! I love it!",
      "Terrible experience, very disappointed."
    ]
  }'
```

## Deployment Workflow

### 1. Build Docker Image

```bash
# Build the transformer image
docker build -t sentiment-transformer:latest .
```

### 2. Tag and Push to Registry

```bash
# Tag image (example using quay.io)
docker tag sentiment-transformer:latest quay.io/your-username/sentiment-transformer:latest

# Push image
docker push quay.io/your-username/sentiment-transformer:latest
```

**Note**: Remember to update the image in [resources/inference_service.yaml](../resources/inference_service.yaml) to match your image.

### 3. Create Kubernetes Resources

First, make sure you have:
- A Kubernetes cluster with KServe installed
- The model stored in accessible storage

Create the test namespace:

```bash
kubectl apply -f transformer-demo
```

Create the MLServer ServingRuntime:

```bash
kubectl apply -f resources/serving-runtime.yaml
```

Then apply the InferenceService:

```bash
# Update the resources/inference_service.yaml with your image registry and storage URIs
kubectl apply -f resources/inference_service.yaml
```

See [resources/inference_service.yaml](../resources/inference_service.yaml) for the complete example.

### 4. Wait for Deployment

```bash
kubectl get inferenceservice sentiment-analysis

# Wait until READY is True
kubectl wait --for=condition=Ready inferenceservice/sentiment-analysis --timeout=300s
```

### 5. Test the Deployed Service

Port forward to the transformer service:
```bash
kubectl port-forward svc/sentiment-analysis-transformer 8080:80
```

In another terminal, test with curl:
```bash
curl -X POST http://localhost:8080/v1/models/sentiment-analysis:predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This product exceeded my expectations!",
      "Worst purchase ever, total waste of money"
    ]
  }'
```