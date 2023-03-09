
```
    # Install this package
    python setup.py build develop
    # need python 3.6 to run ai2thor
    conda create -n waynav python=3.6
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
    pip install ai2thor==2.1.0
    pip install transformers
    pip install torchmetrics tqdm lmdb opencv-python networkx
```