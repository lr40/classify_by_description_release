conda create --name class python
conda activate class
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install cudatoolkit=11.8 -c pytorch
pip install torchmetrics
pip install matplotlib
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch