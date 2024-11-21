<div align="center">
  <h2><img src="assets/logo.png" height="28px"/><i>Spec-Bench:</i> A Comprehensive Benchmark and Unified<br>Evaluation Platform for Speculative Decoding</h2> 
</div>
<p align="center">
| <a href="https://arxiv.org/abs/2401.07851"><b>Paper</b></a> | <a href="https://sites.google.com/view/spec-bench/"><b>Blog</b></a> | <a href="https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md"><b>Leaderboard</b></a> | <a href="ROADMAP.md"><b>Roadmap</b></a> |
</p>





![timeline](./assets/7B.png)

<div align="center">
<font color="gray">Speedup comparison of Speculative Decoding methods on Spec-Bench, evaluated by Vicuna-7B-v1.3.</font>
</div>

> [!TIP]
> Looking for the Token Recycling benchmarks on this branch? Scroll to the bottom of this README.

## Introduction

Spec-Bench is a comprehensive benchmark designed for assessing Speculative Decoding methods across diverse scenarios. Based on Spec-Bench, we aim to establish and maintain a unified evaluation platform for open-source Speculative Decoding approaches. This platform facilitates the systematic assessment of existing methods ***in the same device and testing environment***, thereby ensuring fair comparisons. 

Currently, Spec-Bench supports the evaluation of the following open source models:

- [EAGLE-2](https://github.com/SafeAILab/EAGLE)
- [EAGLE](https://sites.google.com/view/eagle-llm)
- [Hydra](https://github.com/zankner/hydra)
- [Medusa](https://sites.google.com/view/medusa-llm)
- [Speculative Sampling](https://huggingface.co/blog/assisted-generation)
- [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding)
- [REST](https://sites.google.com/view/rest-llm/)
- [Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- [SPACE](https://github.com/cteant/SPACE)

## Update

**2024.10.25**: We have integrated [EAGLE-2](https://github.com/SafeAILab/EAGLE) into Spec-Bench.

**2024.05.29**: We have integrated [SPACE](https://github.com/cteant/SPACE) into Spec-Bench.

**2024.05.16**: Our [paper](https://arxiv.org/abs/2401.07851) has been accepted by ACL 2024 Findings 🎉 !

**2024.03.12**: We now support statistics for [#Mean accepted tokens](https://github.com/hemingkx/Spec-Bench/blob/main/evaluation/speed.py#L65).

**2024.03.11**: We have integrated [Hydra](https://github.com/zankner/hydra) into Spec-Bench, check it out!

## Installation

```
conda create -n specbench python=3.9
conda activate specbench
cd Spec-Bench
pip install -r requirements.txt
```

## Model Weights

Download corresponding model weights (if required) and modify the checkpoint path in `eval.sh`.

- [vicuna-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)
- [EAGLE](https://github.com/SafeAILab/EAGLE?tab=readme-ov-file#eagle-weights)
- [Hydra](https://github.com/zankner/hydra?tab=readme-ov-file#model-weights)
- [Medusa-1](https://github.com/FasterDecoding/Medusa?tab=readme-ov-file#medusa-1)
- [Speculative Sampling](https://github.com/NJUNLP/MCSD?tab=readme-ov-file#model-release)
- [SPACE](https://huggingface.co/AntMan/vicuna-v1.3-7b-space)

## Additonal Setup

#### REST (Optional)

##### Build DraftRetriever from source

```
cd model/rest/DraftRetriever
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
maturin build --release --strip -i python3.9 # will produce a .whl file
pip3 install ./target/wheels/draftretriever-0.1.0-cp39-cp39-linux_x86_64.whl
```

##### Create a datastore

```
cd model/rest/datastore
./datastore.sh # modify your own path
```

## Inference

Select specific command line in `eval.sh`, the results will be stored in `data/spec_bench/model_answer/`.

```
./eval.sh
```

## Speedup Report

Obtain the corresponding speedup compared to vanilla autoregressive decoding.

```
python evaluation/speed.py --file-path /your_own_path/eagle.jsonl --base-path /your_own_path/vicuna.jsonl
```

## Result Comparison

Examine whether the generated results are equal to autoregressive decoding or not.

```
python evaluation/equal.py --file-path /your_own_path/model_answer/ --jsonfile1 vicuna.jsonl --jsonfile2 eagle.jsonl
```

## Contributing

We warmly welcome contributions and discussions related to Spec-Bench! If you have any suggestions for improvements or ideas you'd like to discuss, please don't hesitate to open an issue. This will allow us to collaborate and discuss your ideas in detail.

***More models are welcome!*** - If you're aware of any open-source Speculative Decoding methods not currently included in Spec-Bench, we encourage you to contribute by submitting a pull request. This helps ensure Spec-Bench remains a comprehensive and fair benchmarking platform for comparing existing methods. Please ensure that your changes are well-tested before submission.

## Acknowledgments

This codebase is built from [Medusa](https://github.com/FasterDecoding/Medusa) and [EAGLE](https://github.com/SafeAILab/EAGLE). We integrated code implementations of multiple open-source Speculative Decoding methods to facilitate unified evaluation.

## Citation

If you find the resources in this repository useful, please cite our paper:

```
@inproceedings{xia-etal-2024-unlocking,
    title = "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding",
    author = "Xia, Heming and Yang, Zhe and Dong, Qingxiu and Wang, Peiyi and Li, Yongqi  and Ge, Tao and Liu, Tianyu and Li, Wenjie and Sui, Zhifang",
    editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.456",
    doi = "10.18653/v1/2024.findings-acl.456",
    pages = "7655--7671",
}
```

## Unofficial Token Recycling Benchmarks

- Device: a single NVIDIA A100 GPU (40GB) with 30 CPU cores
- Testing environment: Pytorch 2.5.1, under CUDA 12.4
- Experimental Settings: greedy decoding, FP16 precision, batch size = 1
- Single run (not average of 3 runs like the official leaderboard)
- Cold Start means the Token Recycling adjacency matrix was reset for each prompt.

### Vicuna-7B-v1.3

| Models                                                             | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation | #Mean Accepted Tokens |  Overall  |
| ------------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------------------: | :-------: |
| [Recycling](https://github.com/smpanaro/token-recycling)           | 2.24x                   | 1.87x       | 2.08x          | 1.99x              | 2.50x                  | 1.80x                     | 2.67                  | 2.08x     |
| [Recycling](https://github.com/smpanaro/token-recycling) Cold Start| 2.07x                   | 1.30x       | 2.23x          | 1.70x              | 2.30x                  | 1.95x                     | 2.55                  | 1.93x     |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding)       | 1.56x                   | 1.00x       | 2.54x          | 1.13x              | 1.55x                  | 1.80x                     | 1.75                  | 1.60x     |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) | 1.45x                   | 1.13x       | 1.31x          | 1.20x              | 1.50x                  | 1.16x                     | 1.64                  | 1.30x     |

