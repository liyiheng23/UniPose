#!/bin/bash

# conda config --env --set always_yes true

# conda create -n llava python=3.10
# conda activate posegpt
# conda config --env --remove-key always_yes

pip install ./torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl
pip install ./torchvision-0.16.2+cu121-cp310-cp310-linux_x86_64.whl

cd ..
mkdir submodule 
cd submodule
git clone https://github.com/huggingface/evaluate.git
git clone https://github.com/haotian-liu/LLaVA.git
git clone https://github.com/sha2nkt/moyo_toolkit.git
git clone https://github.com/naver/posescript.git

cd LLaVA
pip install -e .
pip install -e ".[train]"
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip install flash-attn --no-build-isolation

pip install pytorch_lightning==2.3.3
pip install yapf addict spacy webdataset opencv-python ipdb scikit-image yacs ipykernel roma
pip install tensorboard
python -m spacy download en_core_web_sm

# pip install rich six scipy ipdb einops tensorboard evaluate  
# pip install bert_score rouge_score

# llava: transformers==4.37.2
python -m pip install accelerate==0.21.0