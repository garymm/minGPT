# WikiText-2 Language Model with minGPT

This project trains a GPT model (GPT2-XL architecture) on the WikiText-2 dataset, which contains high-quality Wikipedia articles.

## Overview

- Uses the Hugging Face dataset `wikitext` with `wikitext-2-raw-v1` configuration
- Implements a GPT2-XL model (1.5B parameters)
- Includes validation loss tracking and text generation during training

## Usage

Run the training script:

```bash
python projects/wikitext/wikitext.py
```

You can modify various configuration parameters by passing command line arguments, for example:

```bash
python projects/wikitext/wikitext.py --model.model_type=gpt2-medium --trainer.batch_size=16 --trainer.learning_rate=5e-5
```

## Requirements

- PyTorch
- Hugging Face datasets library

## Notes

The GPT2-XL model is quite large and may require significant GPU memory. You might need to adjust the batch size or use a smaller model variant if you encounter memory issues.
