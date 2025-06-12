This repository provides a Python implementation for **semantic data augmentation** of event logs in Predictive Process Monitoring (PPM) using Large Language Models (LLMs). By converting trace prefixes into natural-language prompts and guiding the LLM with candidate activity lists—optionally enriched by process-level statistics—we generate contextually and causally valid synthetic trace fragments to improve next-activity prediction.

---

## ⚙️ Key Features

- **Prompt 1**: Prefix + candidate activities only  
- **Prompt 2**: Prefix + candidate activities + process-level statistics (top start/end)  
- **Batch Inference**: Control batch size for efficient LLM calls  
- **Timestamp Interpolation**: Inserted events receive midpoint timestamps and an `Aug=1` flag  
- **CSV Output**: Export the full augmented log and a summary statistics file  

---

## 🚀 Usage

```bash
python llm_augment.py \
  --input data/Sepsis.csv \
  --output data/Sepsis_augmented.csv \
  --aug_ratio 0.1 \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --batch_size 4 \
  --use_stats
