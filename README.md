<div align="center">

<h1>UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing</h1>

<div>
    <strong>CVPR 2025</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://www.arxiv.org/abs/2411.16781" target='_blank'>
            <img src="https://img.shields.io/badge/arxiv-report-red">
        </a>
        <a href="https://liyiheng23.github.io/UniPose-Page/" target='_blank'>
            <img src="https://img.shields.io/badge/Project-Page-Green">
        </a>
    </h4>
</div>

⭐ If UniPose is helpful to your projects, please help star this repo. Thanks! 🤗

</div>

<be>

<div align="center">
<img src="assets/demo.png" alt="teaser" width="100%">
</div>

This repository contains the official implementation of **UniPose**. UniPose is a multi-modal LLM to understand, generate and edit 3D human poses. It enables seamless transformation of human poses across multi-modalities (visual, text, SMPL) and multi-tasks (comprehension, generation, editing).

## 🔧 Dependencies and Installation

1. Clone this repo and navigate to UniPose folder

   ```bash
   git clone https://github.com/liyiheng23/UniPose.git
   cd UniPose
   ```

2. Create Conda Environment and Install Package

   ```bash
   conda create -n unipose python=3.10 -y
   conda activate unipose
   pip3 install -r requirements.txt
   ```
3. Download Pre-trained Models
      #### Base Model:
      * `CLIP-ViT`: [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
      * `LLaVA`: [liuhaotian/llava-v1.6-mistral-7b](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)
      #### Ours provided Model:


## 🎰 Train
#### I - Prepare training data
Similar to [SeeSR](https://github.com/cswry/SeeSR/blob/main/README.md#step2-prepare-training-data), We pre-prepare HQ-LQ image pairs for the training of IR model. Run the following command to make paired data for training:

```shell
python3 tools/make_paired_data.py \
--gt_path gt_path1 gt_path2 ... \ 
--save_dir /path/to/save/folder/ \
--epoch 1 # number of epochs to generate paired data
```

After generating paired data, you can use MLLM (e.g., [LLaVA](https://github.com/haotian-liu/LLaVA)) to generate detailed text prompt for HQ images. Then you need to use T5 to extract text features in order to save training time. Run:

```shell
python3 tools/extract_t5_features.py \
--t5_ckpt /path/to/t5-v1_1-xxl \
--caption_folder /path/to/caption/folder \
--save_npz_folder /path/to/save/npz/folder
```

Finally, the directory structure for training datasets should look like
```
training_datasets_folder/
    └── gt
        └── 0000001.png # GT , (1024, 1024, 3)
        └── ...
    └── sr_bicubic
        └── 0000001.png # LQ + bicubic upsample, (1024, 1024, 3)
        └── ...
    └── caption
        └── 0000001.txt # Caption files (not used in training)
        └── ...
    └── npz
        └── 0000001.npz # T5 features
        └── ...
```
#### II - Training for DreamClear
Run the following command to train DreamClear with default settings:
```shell
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... \
    train_dreamclear.py configs/DreamClear/DreamClear_Train.py \
    --load_from /path/to/PixArt-XL-2-1024-MS.pth \
    --vae_pretrained /path/to/sd-vae-ft-ema \
    --swinir_pretrained /path/to/general_swinir_v1.ckpt \
    --val_image /path/to/RealLQ250/lq/val_image.png \
    --val_npz /path/to/RealLQ250/npz/val_image.npz \
    --work_dir experiments/train_dreamclear
```
Please modify the path of training datasets in `configs/DreamClear/DreamClear_Train.py`. You can also modify the training hyper-parameters (e.g., `lr`, `train_batch_size`, `gradient_accumulation_steps`) in this file, according to your own GPU machines.
## ⚡ Inference
We provide the `RealLQ250` benchmark, which can be downloaded from [Google Drive](https://drive.google.com/file/d/16uWuJOyGMw5fbXHGcl6GOmxYJb_Szrqe/view?usp=sharing).
#### Testing DreamClear for Image Restoration


Run the following command to restore LQ images (the code defaults to using 2 GPUs for inference):
```shell
python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 \
    test.py configs/DreamClear/DreamClear_Test.py \
    --dreamclear_ckpt /path/to/DreamClear-1024.pth \
    --swinir_ckpt /path/to/general_swinir_v1.ckpt \
    --vae_ckpt /path/to/sd-vae-ft-ema \
    --t5_ckpt /path/to/t5-v1_1-xxl \
    --llava_ckpt /path/to/llava-v1.6-vicuna-13b \
    --lre --cfg_scale 4.5 --color_align wavelet \
    --image_path /path/to/input/images \
    --save_dir validation \
    --mixed_precision fp16 \
    --upscale 4
```
#### Evaluation on high-level benchmarks

Testing instructions for [segmentation](segmentation/README.md) and [detection](detection/README.md) can be found in their respective folders.

## 🪪 License

The provided code and pre-trained weights are licensed under the [Apache 2.0 license](LICENSE).

## 🤗 Acknowledgement

This code is based on [ChatPose](https://github.com/yfeng95/PoseGPT), [LISA](https://github.com/dvlab-research/LISA), [LLaVA](https://github.com/haotian-liu/LLaVA) and [PoseScript](https://github.com/naver/posescript). Some code are brought from [TokenHMR](https://github.com/saidwivedi/TokenHMR), [4D-Humans](https://github.com/shubham-goel/4D-Humans). We thank the authors for their awesome work.

## 📧 Contact
If you have any questions, please feel free to reach me out at liyiheng23@gmail.com. 

## 📖 Citation
If you find our work useful for your research, please consider citing our paper:
```
@article{li2024unipose,
  title={UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing},
  author={Li, Yiheng and Hou, Ruibing and Chang, Hong and Shan, Shiguang and Chen, Xilin},
  journal={arXiv preprint arXiv:2411.16781},
  year={2024}
}
```
