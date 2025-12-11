# Image_Classification_with_LLM_Structure

## Overview
An image classification project leveraging the Transformer architecture from Large Language Models (LLMs).

## Key Features
- **Multiple LLM Architectures**: BERT, ELECTRA, GPT-2, RoBERTa, T5
- **Diverse Feature Extractors**: VGG, ResNet, DenseNet, EfficientNet, MobileNet, ConvNeXt
- **Multi-Scale Feature Extraction**: Extract image features at various scales for enhanced performance

## Project Structure
```
├── models/
│   ├── bert.py                # BERT-based image classification model
│   ├── electra.py             # ELECTRA-based image classification model
│   ├── gpt.py                 # GPT-2-based image classification model
│   ├── roberta.py             # RoBERTa-based image classification model
│   ├── t5.py                  # T5-based image classification model
│   └── feature_extractor.py  # CNN-based feature extractors
```

## How It Works
1. Extract image features using CNN-based Feature Extractors
2. Project extracted features into Transformer embedding dimensions
3. Process features through LLM Transformer architecture
4. Perform final classification through MLP layers
