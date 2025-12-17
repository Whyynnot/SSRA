# README
Codes for "Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling".

Our paper is available at: https://arxiv.org/abs/2509.16717

## Training Model

```bash
bash train_qwen/run_train.sh
```

## Evaluation

### Retrieval Task
```bash
bash eval/eval_retrieval/run_eval.sh
```

### Pair-Classification Task
```bash
bash eval/eval_pair-classification/run_eval.sh
```

## Citation

If you find SSRA useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{li2025semi,
  title={Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling},
  author={Li, Haoran and Su, Zhiming and Yao, Junyan and Zhang, Enwei and Ji, Yang and Chen, Yan and Zhou, Kan and Feng, Chao and Ran, Jiao},
  journal={arXiv preprint arXiv:2509.16717},
  year={2025}
}
```

## Acknowledgement

The codebase of SSRA references the following open-source projects, and we thank them for their contributions:

* The overall code framework references [Piccolo2](https://github.com/OpenSenseNova/piccolo-embedding).
* Some code files referenced these projects: [LAVIS](https://github.com/salesforce/LAVIS), [MoCo](https://github.com/facebookresearch/moco), [Inf-CLIP](https://github.com/DAMO-NLP-SG/Inf-CLIP).

## License

This project is licensed under Apache2.0. See the [LICENSE](LICENSE). flie for details.
