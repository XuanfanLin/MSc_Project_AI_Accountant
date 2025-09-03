# MSc_Project_AI_Accountant

## Environment Setup

This project was developed in **Python 3.9**.

### 1. Create and activate a virtual environment
```bash
python3 -m venv ai_accountant_env
source ai_accountant_env/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. GPU support
The environment has been tested on NVIDIA A100 (CUDA 12.6).
All experiments were conducted on GPU servers provided by the UCL EEE Department.

---

## Project Structure

```
.
├── data/
│   └── Put_uk_tax_synthetic_dataset.jsonl_data_here.txt   # dataset placeholder
│
├── models/                                                # fine-tuned 
│   ├──lora-sft/
│   ├──tpo_1/
│   ├──tpo_2/
│
├── scripts/
│   ├── 1.data_preparation/                                # data preparation
│   │   ├── eda/
│   │   ├── uk_tax_law_extraction/
│   │   ├── urls/
│   │   ├── eda.py
│   │   ├── hmrc_scraper.py
│   │   ├── legislation_scraper.py
│   │   └── split_dataset.py
│   │
│   ├── 2.sft/                                             # supervised fine-tuning
│   │   ├── download_Llama3.py
│   │   └── sft.py
│   │
│   ├── 3.tpo/                                             # Thought Preference Optimization (TPO)
│   │   ├── generate_thought_response_pairs.py
│   │   ├── germini_rank_judge.py
│   │   ├── run_dpo.py
│   │   └── test.py
│   │
│   ├── 4.tpo2/                                            # iterative TPO (TPO2)
│   │   ├── output/
│   │   │   └── Put_dpo_pairs_gpt_enhanced.jsonl_dataset_here.txt
│   │   ├── generate_thought_response_pairs.py
│   │   ├── germini_rank_judge.py
│   │   └── run_dpo.py
│   │
│   └── 5.overall_benchmark/                               # benchmarking & evaluation
│       ├── analyse_score_overall.py
│       ├── benchmark_for_all.py
│       ├── benchmark_result_is_here.txt
│       ├── score_dist_box_clean.png
│       ├── score_ecdf.png
│       ├── score_survival.png
│       └── summary_scores.json
│
├── run_tpo2_thought_response.py                           # final evaluation script
├── requirements.txt
└── README.md

```

---

## Execution Pipeline

1. **Data Preparation**  
   - Run `hmrc_scraper.py`, `legislation_scraper.py`.  
   - Download dataset from Kaggle:  
     👉 [UK Tax Synthetic Dataset](https://www.kaggle.com/datasets/xuanfanlin/uk-tax-synthetic-dataset)  
     Put it into: `data/uk_tax_synthetic_dataset.jsonl`  
   - Run `split_dataset.py`, then `eda.py`.

2. **Supervised Fine-Tuning (SFT)**  
   - Run `download_Llama3.py`  
   - Run `sft.py`

3. **TPO (first iteration)**  
   - Run `generate_thought_response_pairs.py`  
   - Run `germini_rank_judge.py`  
   - Run `run_dpo.py`

4. **TPO2 (second iteration)**  
   - Run `generate_thought_response_pairs.py`  
   - Run `germini_rank_judge.py`  
   - Download dataset from Kaggle:  
     👉 [DPO Pairs GPT Enhanced](https://www.kaggle.com/datasets/xuanfanlin/dpo-pairs-gpt-enhanced)  
     Put it into: `scripts/4.tpo2/output/dpo_pairs_gpt_enhanced.jsonl`  
   - Run `run_dpo.py`

5. **Overall Benchmark**  
   - Run `benchmark_for_all.py`  
   - Run `analyse_score_overall.py`

6. **Final Model Test**  
   - Run `run_tpo2_thought_response.py`

---

## Reference (Abstract)

In the United Kingdom, access to reliable tax advice is often limited by the high cost of professional services and the complexity of legislation. Recent advances in large language models (LLMs) offer opportunities to democratise expert-level guidance. This research introduces **The AI Accountant**, a domain-adapted advisory system built on *Meta-Llama-3-8B*. The system is trained through a two-stage pipeline: (i) *supervised fine-tuning (SFT)* on 1,000 synthetic instruction–response pairs grounded in HMRC manuals and UK legislation, and (ii) *Thought Preference Optimisation (TPO)*, which combines preference signals from Gemini 2.0 with Direct Preference Optimisation (DPO) to enhance reasoning quality, factual reliability, and stability.  

Experiments on a held-out test set show marked improvements: mean Gemini scores rise from **32.7 (Base) → 59.0 (SFT)**, with further stabilisation through iterative **TPO (61.3 at TPO2)**. These results demonstrate that relatively small open-source models, once fine-tuned on domain-specific data, can provide satisfactory performance in the specialised context of UK taxation.  

While high-end performance remains limited, the pipeline reduces low-quality outputs, strengthens mid-range reliability, and embeds interpretability via structured reasoning traces. Overall, the study highlights the feasibility of combining open-source models with domain adaptation to support compliant, explainable, and accessible tax advisory systems, paving the way for integration into taxpayer self-service platforms and digital accounting tools.  

