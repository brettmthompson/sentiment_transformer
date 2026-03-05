# KServe Sentiment Analysis Transformer

This demo shows how to use a KServe combined transformer (preprocessing and postprocessing) with the MLServer runtime. While the code supports a variety of models and model formats, the demo utilizes the HuggingFace Optimum `distilbert-base-uncased-finetuned-sst-2-english` sentiment analysis model (ONNX-optimized).

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Components](#components)
- [Installation](#installation)
- [Usage and Deployment](#usage-and-deployment)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Demo Model Information](#demo-model-information)
- [References](#references)

## Features

- **Combined Transformer**: Single component handling both pre and postprocessing
- **Protocol Support**:
  - V1 protocol with custom JSON for client requests (`{"texts": [...]}`)
  - V2 inference protocol for predictor communication
- **Flexible Configuration**: Customizable sentiment labels, input/output names, and sequence length
- **Star Ratings**: Optional 1-5 star rating calculation for sentiment (negative/positive) via `--include_star_rating` flag

## Architecture

```
Client Request (V1 Protocol - JSON with text)
    ↓
Transformer (transformer.py)
    Preprocess:
      - Validates input
      - Tokenizes text using HuggingFace tokenizer
      - Formats tokens for ONNX model request (V2 protocol)
    ↓
Predictor (MLServer with ONNX runtime)
    - Runs inference on ONNX model
    - Returns raw logits (V2 protocol)
    ↓
Transformer (transformer.py)
    Postprocess:
      - Handles ONNX model response (V2 Protocol)
      - Applies softmax to logits
      - Maps to sentiment labels
      - Formats response with confidence scores
      - Optionally adds star rating based on labels (when enabled)
    ↓
Client Response (V1 Protocol - JSON with sentiment)
```

## Components

### Combined Transformer (`transformer.py`)
A single KServe transformer that handles both preprocessing and postprocessing. It accepts custom JSON requests over the V1 protocol from clients and communicates with the predictor using the V2 inference protocol.

**Preprocessing**:
- **Input**: Custom JSON format `{"texts": ["..."]}` (from client)
- **Processing**:
  - Validates and extracts text from payload
  - Tokenizes using HuggingFace `AutoTokenizer`
  - Configurable input names (defaults to `input_ids`, `attention_mask`)
  - Handles padding and truncation
  - Converts to `InferRequest` (V2 protocol for predictor)
- **Output**: `InferRequest` with tokenized inputs → sent to predictor

**Postprocessing**:
- **Input**: `InferResponse` with raw model logits (V2 protocol from predictor)
- **Processing**:
  - Extracts output data from `InferResponse`
  - Applies softmax for probability conversion
  - Determines predicted sentiment class
  - Calculates confidence scores for all classes
  - Optionally generates 1-5 star rating (when `--include_star_rating` flag is set and using negative/positive labels)
- **Output**: Custom JSON with sentiment label, confidence, all class scores, and optionally star rating → returned to client

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and the project package
uv sync
```

**Note**: This installs **both dependencies and the project package** with the standard PyTorch package including GPU dependencies (~2-4GB). For CPU-only environments, see the pip alternative below.

### Using pip (alternative - CPU-only PyTorch)

For a lighter installation using CPU-only PyTorch (~150-200MB):

```bash
pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

**Note**: This installs **only the dependencies** (not the project package itself) using lightweight CPU-only PyTorch. This is useful for containerized deployments where you only need the dependencies installed, and the application code is mounted separately. For local development, use `uv sync` instead.

## Usage and Deployment

For instructions on running locally and deploying to KServe, see the [Quick Start Guide](docs/QUICKSTART.md).

The guide covers:
- Local testing without KServe
- Running the development server
- Building and pushing container images
- Deploying to Kubernetes with KServe
- Testing deployed services

## Customization

### Adjusting Sentiment Labels
Modify the `--sentiment_labels` argument to align with your model or limit those that are processed:
```bash
--sentiment_labels=very_negative,negative,neutral,positive,very_positive
```

**Note**: Star ratings (1-5) are only calculated when the `--include_star_rating` flag is set and `negative,positive` labels are used. For other label configurations or when the flag is not set, only sentiment and confidence scores are returned.

### Changing Max Sequence Length
Adjust the `--max_length` argument:
```bash
--max_length=256
```

### Input Names
Adjust input names on a per model basis (deafults are positive and negative):
```bash
--input_names=input_ids,attention_mask,token_type_ids
```

### Output Name
If your model or runtime uses a different output name than "predict":
```bash
--output_name=output
```

### Star Rating
To include 1-5 star ratings in the response payload (only works with negative/positive labels):
```bash
--include_star_rating
```

By default, star ratings are **not** included. When enabled, the transformer will add a `star_rating` field to each prediction based on the sentiment and confidence:
- **5 stars**: Positive with ≥80% confidence
- **4 stars**: Positive with <80% confidence
- **3 stars**: Uncertain (40-60% confidence either way)
- **2 stars**: Negative with <80% confidence
- **1 star**: Negative with ≥80% confidence

## Troubleshooting

### Common Issues

1. **Tokenizer not found**: Ensure you have internet connectivity to download from HuggingFace and the tokenizer name is correct for your model
   - **For private tokenizers**: Set the `HF_TOKEN` environment variable with your HuggingFace access token:
     ```bash
     export HF_TOKEN=your_huggingface_token_here
     ```
2. **Shape mismatches**: Check that the model's expected input names match what the transformer is sending. You can adjust the `--input_names` argument to match your model's expected inputs
3. **Missing predictions**: The transformer expects an `InferResponse` with V2 protocol format. If the predictor returns a different format, check:
   - The predictor's runtime
   - The predictor and runtime use V2 protocol
   - The output name matches what the transformer expects (default: "predict")

## Demo Model Information

- **Model**: HuggingFace Optimum DistilBERT for Sentiment Analysis
- **HuggingFace Hub**: `optimum/distilbert-base-uncased-finetuned-sst-2-english`
- **Accuracy**: 91.3% on SST-2 dev set
- **Model Size**: 268 MB (ONNX FP32, officially exported via HuggingFace Optimum)
- **Classes**: Binary sentiment (negative, positive)

## References

- [KServe Custom Transformers](https://kserve.github.io/website/docs/model-serving/predictive-inference/transformers/custom-transformer)
- [HuggingFace Optimum DistilBERT ONNX Model](https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english)
- [Base DistilBERT SST-2 Model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [HuggingFace Optimum Documentation](https://huggingface.co/docs/optimum)
- [MLServer Documentation](https://mlserver.readthedocs.io/)
