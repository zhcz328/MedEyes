# MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis

###  🌟 Abstract
Accurate medical diagnosis often involves progressive visual focusing and iterative reasoning, characteristics commonly observed in clinical workflows. While recent vision-language models demonstrate promising chain-of-thought (CoT) reasoning capabilities via reinforcement learning with verifiable rewards (RLVR), their purely on-policy learning paradigm tends to reinforce superficially coherent but clinically inaccurate reasoning paths. We propose MedEyes, a novel reinforcement learning framework that dynamically models clinician-style diagnostic reasoning by progressively attending to and interpreting relevant medical image regions. By incorporating off-policy expert guidance, MedEyes converts expert visual search trajectories into structured external behavioral signals, guiding the model toward clinically aligned visual reasoning. We design the Gaze-guided Reasoning Navigator (GRN) to emulate the diagnostic process through a dual-mode exploration strategy, scanning for systematic abnormality localization and drilling for detailed regional analysis. To balance expert imitation and autonomous discovery, we introduce the Confidence Value Sampler (CVS), which employs nucleus sampling and adaptive termination to create diverse yet credible exploration paths. Finally, the dual-stream GRPO optimization framework decouples on-policy and off-policy learning signals, mitigating reward assimilation and entropy collapse. Experiments demonstrate that MedEyes achieves an average performance improvement of +8.5\% across multiple medical VQA benchmarks, validating MedEyes's potential in building interpretable medical AI systems.

![PDF预览](Figure/medeyes.pdf)
## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0
- See `requirements.txt` for full dependencies

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MedEyes.git
cd MedEyes

# Create virtual environment
conda create -n medeyes python=3.8
conda activate medeyes

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
bash scripts/download_models.sh

## 📊 Datasets

### Supported Datasets

- VQA-RAD
- SLAKE
- PathVQA
- PMC-VQA
- MMMU (Health & Medicine)

### Data Preparation

```bash
# Download and prepare datasets
python scripts/prepare_data.py --dataset vqa-rad --output_dir ./data

# Organize data structure
data/
├── vqa-rad/
│   ├── images/
│   ├── annotations/
│   └── splits/
├── slake/
└── ...
```

## 🔧 Training

### Basic Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --dataset vqa-rad \
    --output_dir ./outputs
```

### Distributed Training

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
    scripts/train.py \
    --config configs/default.yaml \
    --distributed
```

## 📈 Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/checkpoint_best.pth \
    --dataset vqa-rad \
    --split test
```

## 🎯 Demo

```bash
python scripts/demo.py \
    --checkpoint ./outputs/checkpoint_best.pth \
    --image_path ./examples/chest_xray.jpg \
    --question "Is there evidence of pneumothorax?"
```

## 📖 Architecture

### Gaze-guided Reasoning Navigator (GRN)

- **Scanning Mode**
- **Drilling Mode**
- **Mode Transition**

### Confidence Value Sampler (CVS)

- **Nucleus Sampling**
- **Dynamic Termination**
- **Trajectory Parsing**

### Dual-stream GRPO

- **On-policy Stream**
- **Off-policy Stream**
- **Advantage Decoupling**

[//]: # (## 📝 Citation)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{medeyes2025,)

[//]: # (  title={MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis},)

[//]: # (  author={Anonymous},)

[//]: # (  journal={arXiv preprint arXiv:2025.xxxxx},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

## 🤝 Acknowledgments

This project builds upon:

- [MedPLIB](https://github.com/ShawnHuang497/MedPLIB)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen-VL)
## Code Structure

```
MedEyes/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── default.yaml
│   └── dataset_configs/
│       ├── vqa_rad.yaml
│       ├── slake.yaml
│       └── pathvqa.yaml
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── medical_vqa_dataset.py
│   └── data_utils.py
├── models/
│   ├── __init__.py
│   ├── grn.py                   
│   ├── cvs.py                   
│   ├── medeyes.py               
│   ├── medplib_integration.py   
│   └── qwen_vl_wrapper.py      
├── training/
│   ├── __init__.py
│   ├── dual_stream_grpo.py      
│   ├── replay_buffer.py        
│   ├── rewards.py              
│   └── trainer.py               
├── inference/
│   ├── __init__.py
│   ├── predictor.py
│   └── visualization.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── prompt_generator.py
│   └── tool_utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
└── experiments/
    └── run_experiments.sh
```

## 
