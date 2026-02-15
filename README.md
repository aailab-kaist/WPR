# Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment (WPR) (ICLR 2026)

| [paper](https://arxiv.org/abs/2602.01685) | [openreview](https://openreview.net/forum?id=sUac3QDbAs) | [code](https://github.com/aailab-kaist/WPR) |

--------------------

This repository contains the official implementation of **"Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment"** in **[ICLR 2026](https://iclr.cc/Conferences/2026)**.

**[Byeonghu Na](https://sites.google.com/view/byeonghu-na), [Hyungho Na](https://sites.google.com/view/asd-lab), [Yeongmin Kim](https://sites.google.com/view/yeongmin-space/), [Suhyeon Jo](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=10), [HeeSun Bae](https://sites.google.com/view/baeheesun), [Mina Kang](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=25), and [Il-Chul Moon](https://aai.kaist.ac.kr)**

**KAIST, UNIST, summary.ai**

--------------------

**Wasserstein Policy Regularization (WPR)** is a semantic-aware regularization for the reinforcement learning from human feedback (RLHF) framework based on the entropy-regularized Wasserstein distance, which incorporates the geometry of the token space.

<img src="./assets/overview.png" width="1000" title="overview">

--------------------

## Requirements

We utilized:

- CUDA 11.4
- Python 3.8
- NVIDIA A100 GPUs
- DeepSpeed-Chat Framework

### Installation

```bash
export PYTHONPATH=${WPR_BASE}/applications/DeepSpeed-Chat:$PYTHONPATH

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

cd ${WPR_BASE}/applications/DeepSpeed-Chat
pip install -e .
```

## Datasets

We conduct experiments on [TL;DR](https://huggingface.co/datasets/openai/summarize_from_feedback), [HH-RLHF](https://huggingface.co/datasets/Dahoas/full-hh-rlhf), and [APPS](https://huggingface.co/datasets/codeparrot/apps) datasets, following the same setup as [MA-RLHF](https://github.com/ernie-research/MA-RLHF?tab=readme-ov-file#dataset).

Our codebase is built upon MA-RLHF, and we provide an extended implementation for the APPS dataset, enabling alignment experiments on code-generation tasks as well.
All required datasets will be automatically downloaded and prepared when running the training scripts.

## RLHF Training Pipeline

The overall RLHF training pipeline is consistent with the base framework [MA-RLHF](https://github.com/ernie-research/MA-RLHF?tab=readme-ov-file#training).
For detailed argument descriptions and additional configurations, please refer to the [MA-RLHF](https://github.com/ernie-research/MA-RLHF?tab=readme-ov-file#training) and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) documentation.

WPR is applied during Step 3 (PPO Training) as a policy regularization term.

Below, we provide example commands for training on the TL;DR dataset.


### Step 1: Supervised Fine-Tuning (SFT)

```bash
cd applications/DeepSpeed-Chat/training/step1_supervised_finetuning

deepspeed -i localhost:0,1,2,3,4,5,6,7 --master_port 1234 main.py \
  --data_path openai/summarize_from_feedback \
  --data_split 2,4,4 \
  --model_name_or_path google/gemma-2b \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --max_seq_len 1024 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --seed 1234 \
  --gradient_checkpointing \
  --zero_stage 2 \
  --deepspeed \
  --output_dir ../../models/summarize/sft
```


### Step 2: Reward Model Training

```bash
cd applications/DeepSpeed-Chat/training/step2_reward_model_finetuning

deepspeed -i localhost:0,1,2,3,4,5,6,7 --master_port 1234 main.py \
  --data_path openai/summarize_from_feedback \
  --data_split 2,4,4 \
  --model_name_or_path ../../models/summarize/sft \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --max_seq_len 1024 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --num_padding_at_beginning 0 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --seed 1234 \
  --gradient_checkpointing \
  --zero_stage 2 \
  --deepspeed \
  --output_dir ../../models/summarize/reward_model
```


### Step 3: PPO Training with WPR

```bash
cd applications/DeepSpeed-Chat/training/step3_rlhf_finetuning

deepspeed -i localhost:0,1,2,3,4,5,6,7 --master_port 1234 main_wpr.py \
  --data_path openai/summarize_from_feedback \
  --data_split 2,4,4 \
  --actor_model_name_or_path ../../models/summarize/sft \
  --critic_model_name_or_path ../../models/summarize/reward_model/step_581 \
  --num_padding_at_beginning 0 \
  --per_device_generation_batch_size 8 \
  --per_device_training_batch_size 8 \
  --generation_batches 1 \
  --ppo_epochs 1 \
  --max_answer_seq_len 256 \
  --max_prompt_seq_len 512 \
  --actor_learning_rate 1.5e-5 \
  --critic_learning_rate 1.5e-5 \
  --actor_weight_decay 0.1 \
  --critic_weight_decay 0.1 \
  --num_train_epochs 1 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type linear \
  --gradient_accumulation_steps 4 \
  --actor_gradient_checkpointing \
  --critic_gradient_checkpointing \
  --end_of_conversation_token "<eos>" \
  --actor_dropout 0.0 \
  --deepspeed \
  --seed 1234 \
  --actor_zero_stage 2 \
  --critic_zero_stage 3 \
  --offload \
  --kl_ctl 0.05 \
  --temperature 0.8 \
  --gamma 1.0 \
  --lam 0.95 \
  --termination_condition fixed \
  --ngram 1 \
  --value_function equal \
  --max_iter 10 \
  --cost_dim 512 \
  --top_k 128 \
  --output_dir ../../models/summarize/wpr
```


#### Key Arguments

- `--actor_model_name_or_path`: path to the SFT checkpoint
- `--critic_model_name_or_path`: path to the reward model checkpoint
- `--per_device_training_batch_size`: training batch size per GPU
- `--kl_ctl`: policy regularization hyperparameter ($\beta$)
- `--max_iter`: number of Sinkhorn iterations
- `--cost_dim`: truncation hyperparameter $k_1$ (nearest-$k_1$)
- `--top_k`: truncation hyperparameter $k_2$ (top-$k_2$)


#### Other Step 3 RL Training

* For standard KL-regularized PPO training, we use `main.py`, following the original MA-RLHF implementation.

* Since the **APPS** dataset does not rely on a reward model, slight modifications are required.
Therefore, we provide separate scripts:
  - **KL-regularized PPO for APPS**: `main_apps.py`  
  - **WPR-based PPO for APPS**: `main_wpr_apps.py`


## Inference

Our inference and evaluation scripts are directly adapted from [MA-RLHF](https://github.com/ernie-research/MA-RLHF?tab=readme-ov-file#inference).

Run inference and store generated outputs with reward scores:

```bash
python inference/inference_with_rewards.py \
  --proj-path applications/DeepSpeed-Chat \
  --dataset summarize \
  --model applications/DeepSpeed-Chat/models/summarize/${MODEL_NAME}/step-${STEP} \
  --reward-model applications/DeepSpeed-Chat/models/summarize/${REWARD_MODEL_NAME}/step_${REWARD_MODEL_STEP} \
  --temperature 0.8 \
  --gpu 0 \
  --batch-size ${BATCH_SIZE}
```


## Evaluation

This section provides utilities for GPT-4 based pairwise evaluation.

### 1) Sample from Dataset

 we randomly select 50 instances from the inference results.
 
```bash
python evaluation/tools/sample_from_dataset.py \
  --filepath applications/DeepSpeed-Chat/results/summarize/temperature=0.8/${MODEL_NAME}_step-${STEP}.jsonl \
  --savepath applications/DeepSpeed-Chat/results/summarize/temperature=0.8/${MODEL_NAME}_step-${STEP}-sampled.jsonl \
  --dataset summarize
```

### 2) GPT-4 Pairwise Evaluation

```bash
python evaluation/gpt4-eval.py \
  --sk <OPENAI_API_KEY> \
  --model_name_a applications/DeepSpeed-Chat/results/summarize/temperature=${temperature}/${MODEL_NAME_A}_step-${STEP_A}-sampled.jsonl \
  --model_name_b applications/DeepSpeed-Chat/results/summarize/temperature=${temperature}/${MODEL_NAME_B}_step-${STEP_B}-sampled.jsonl \
  --dataset summarize \
  --output applications/DeepSpeed-Chat/results/summarize/temperature=${temperature}/${MODEL_NAME_A}_step-${STEP_A}-v.s.-${MODEL_NAME_B}_step-${STEP_B}.jsonl
```

## Acknowledgements

This codebase builds upon and is inspired by:

- **MA-RLHF**: https://github.com/ernie-research/MA-RLHF
- **DeepSpeed**: https://github.com/huggingface/diffusers


## Citation

```bibtex
@inproceedings{
na2026semanticaware,
title={Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment},
author={Byeonghu Na and Hyungho Na and Yeongmin Kim and Suhyeon Jo and HeeSun Bae and Mina Kang and Il-chul Moon},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=sUac3QDbAs}
}
```
