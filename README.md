---
title: Southern Newswires
description: Pipeline for identifying and classifying Southern newswire articles.
layout: default
---

# Southern Newswires Project

## Executive Summary
This documentation provides a pipeline for identifying and classifying newswire articles from Southern newspapers (1960–1975). It uses YOLO-based layout detection and BERT classifiers for wire service identification (`AP`, `UPI`, `NEA`). This is part of a larger newspaper article extraction project which covers the entire United States including locally produced articles which are out of copyright. 

### Key Contributions:
- **Open-Source Newswire Classifiers:** Three BERT models for `AP`, `UPI`, and `NEA`, available on [Hugging Face](https://huggingface.co/mikemcrae25/newswire_classifiers).
- **Newspaper layout parser:** *Coming soon.*
- **Labeled Newswire Dataset:** *Coming soon.*
- **Reproducible Research Scripts:** *Coming soon.*

---

## Overview of Pipeline Components
1. **Layout Parsing:** YOLOv10 model segments articles, headlines, ads, comics, masthead, etc.
2. **Newswire Classification:** BERT models identify wire services from headlines and bylines.
3. **Duplication Classificaion:** Bi-encoder identifies duplciations of the same underlying wire dispatch. 
4. **Text Correction:** Llama3.2 corrects common OCR errors.

---

## Newswire Classifiers and Performance
Three BERT-based classifiers (`AP`, `UPI`, `NEA`) were trained on 4,000 labeled articles each (1,000 per wire service, 3,000 from other sources). Training used headlines and the first 100 characters of articles and ran on a TPU (v2-8) over 4 epochs.

**Metrics from Hugging Face Models:**
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|----------|-------|----------|
| **AP** | 0.9925 | 0.9926 | 0.9925 | 0.9925 |
| **UPI** | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| **NEA** | 0.9875 | 0.9880 | 0.9875 | 0.9876 |

**Hugging Face Models:**
- **AP Classifier:** [AP Model](https://huggingface.co/mikemcrae25/newswire_classifiers/ap)
- **UPI Classifier:** [UPI Model](https://huggingface.co/mikemcrae25/newswire_classifiers/upi)
- **NEA Classifier:** [NEA Model](https://huggingface.co/mikemcrae25/newswire_classifiers/nea)

---

## Usage Instructions
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load AP Classifier
tokenizer = AutoTokenizer.from_pretrained('mikemcrae25/newswire_classifiers/ap')
model = AutoModelForSequenceClassification.from_pretrained('mikemcrae25/newswire_classifiers/ap')

article = "(AP) President addresses the nation..."
inputs = tokenizer(article, return_tensors="pt")
outputs = model(**inputs)
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
label = torch.argmax(probs).item()
print("AP Newswire Classification:", label)
```

---

## Availability Status
- **Dataset and Scripts:** *Coming soon.*
- **Models:** Available on Hugging Face.

## References
- Dell et al. (2024). *American Stories Project.*
- Hugging Face: [Newswire Classifiers](https://huggingface.co/mikemcrae25/newswire_classifiers)

---


