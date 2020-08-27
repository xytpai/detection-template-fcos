### DetX-FCOS

Implementation of FCOS in PyTorch. <br>
FCOS: Fully Convolutional One-Stage Object Detection. <br>
https://arxiv.org/abs/1904.01355

#### Usage

```txt
1. Install (PyTorch >= 1.0.0)
sh install.sh

2. Training COCO 1x
python tools/train.py --cfg configs/fcos_r50_sq1025_1x.yaml

3. COCO Eval
python tools/eval_mscoco.py --cfg configs/fcos_r50_sq1025_1x.yaml

4. Demo
python tools/demo.py --cfg configs/fcos_r50_sq1025_1x.yaml
```
