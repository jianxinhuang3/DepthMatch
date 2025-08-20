# DepthMatch

This codebase contains the official PyTorch implementation of <b>DepthMatch</b>:

> **[DepthMatch: Semi-Supervised RGB-D Scene Parsing through Depth-Guided Regularization](https://ieeexplore.ieee.org/abstract/document/11020785)**</br>
> Jianxin Huang, Jiahang Li, Sergey Vityazev, Alexander Dvorkovich, Rui Fan</br>
> IEEE Signal Processing Letters, 2025


## Getting Started

### Pre-trained Encoders

[DINOv2-Small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)

```
├── ./pretrained
    └── dinov2_small.pth
```

### Datasets

- [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

- [KITTI Segmentation](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)


## Training


```bash
sh scripts/train_depthmatch.sh <num_gpu> <port> <split>
# to fully reproduce our results, the <num_gpu> should be set as 1
# otherwise, you need to adjust the learning rate accordingly
```



## Citation

If you find this project useful, please consider citing:

```bibtex
@article{huang2025depthmatch,
  title={DepthMatch: Semi-Supervised RGB-D Scene Parsing through Depth-Guided Regularization},
  author={Huang, Jianxin and Li, Jiahang and Vityazev, Sergey and Dvorkovich, Alexander and Fan, Rui},
  journal={IEEE Signal Processing Letters},
  year={2025},
  publisher={IEEE}
}
```

## Acknowlegement
_**DepthMatch**_ is built upon [UniMatch V2](https://github.com/LiheYoung/UniMatch-V2). We thank the authors for their excellent work and for making the source code publicly available.
