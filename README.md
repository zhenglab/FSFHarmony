<base target="_blank"/>

# FSFHarmony: Towards Spatio-Temporal Consistency for Video Harmonization via Frequency-Spatial Fusion

Here we provide the PyTorch implementation and pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train/Test
- Download [HYouTube](https://github.com/bcmi/Video-Harmonization-Dataset-HYouTube) dataset.

- Train our FSFHarmony model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model fssingleout --netG s_apswin_t_sgt_f3Dfourier --name experiment_name --dataset_root <dataset_dir> --dataset_mode hytall --batch_size xx --init_port xxxx --n_frames 5 --loss_T --loss_AP --save_iter_model
```
- Test our FSFHarmony model:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model fssingleout --netG s_apswin_t_sgt_f3Dfourier --name experiment_name --dataset_root <dataset_dir> --dataset_mode hytall --batch_size 1 --init_port xxxx --n_frames 20
```

## Apply a pre-trained model
- Download pre-trained models from [BaiduCloud](https://pan.baidu.com/s/1l4x-sOEwhCt6KwSOI5hnXg?pwd=p0k6) (access code: p0k6), and put `latest_net_G.pth` in the directory `checkpoints/s_apswin_t_sgt_f3Dfourier`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model fssingleout --netG s_apswin_t_sgt_f3Dfourier --name s_apswin_t_sgt_f3Dfourier --dataset_root <dataset_dir> --dataset_mode hytall --batch_size 1 --init_port xxxx --n_frames 20
```
## Evaluation
To evaluate the spatial consistency, run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root results/experiment_name/test_latest/images/ --evaluation_type hyt --dataset_name HYT
```
To evaluate the temporal consistency, run:
```bash
python all_tc_evaluation.py --dataset_root <dataset_dir> --experiment_name experiment_name --mode 'v' --brightness_region 'foreground'
```
