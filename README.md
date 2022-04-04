# ImpliCity: City Modeling from Satellite Images with Deep Implicit Occupancy Fields

![ImpliCity](docs/teaser.jpg?raw=true)

This repository provides the code to train and evaluate ImpliCity: a method that reconstructs digital surface models 
(DSMs) from raw photogrammetric 3D point clouds and ortho-images with the help of an implicit neural 3D scene representation.
It represents the official implementation of the paper:

### [ImpliCity: City Modeling from Satellite Images with Deep Implicit Occupancy Fields](https://arxiv.org/abs/2201.09968)
Corinne Stucker, Bingxin Ke, Yuanwen Yue, Shengyu Huang, Iro Armeni, Konrad Schindler

> Abstract: *High-resolution optical satellite sensors, combined with dense stereo algorithms, have made it possible to reconstruct 
3D city models from space. However, these models are, in practice, rather noisy and tend to miss small geometric features that are 
clearly visible in the images. We argue that one reason for the limited quality may be a too early, heuristic reduction of the 
triangulated 3D point cloud to an explicit height field or surface mesh. To make full use of the point cloud and the underlying images, 
we introduce ImpliCity, a neural representation of the 3D scene as an implicit, continuous occupancy field, driven by learned embeddings 
of the point cloud and a stereo pair of ortho-photos. We show that this representation enables the extraction of high-quality DSMs: 
with image resolution 0.5m, ImpliCity reaches a median height error of ≈0.7 m and outperforms competing methods, especially w.r.t. 
building reconstruction, featuring intricate roof details, smooth surfaces, and straight, regular outlines.*


## Requirements
The code has been developed and tested with:
* Ubuntu 20.04.4 LTS
* Python 3.8.10
* PyTorch 1.10.0
* CUDA 10.2


## Installation


To create a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) and install the required dependencies, please run:

```bash
git clone git@github.com:prs-eth/ImpliCity.git
cd ImpliCity
python3 -m venv venv/implicity_env
source venv/implicity_env/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Manual installation of torch-scatter:**

To install the binaries for PyTorch 1.10.0, simply run:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
```
where `${CUDA}` should be replaced by either `cpu`, `cu102` (our environment), `cu111`, or `cu113`, depending on your PyTorch installation.

For other major OS/PyTorch/CUDA combinations, please refer to torch-scatter [official documentation](https://github.com/rusty1s/pytorch_scatter#pytorch-1100).


## Usage
### Training
To train ImpliCity from scratch, run:

```bash
python train.py <path/to/config.yaml>
```

For available training options, take a look at the example configuration files:
* `./config/train_test/ImpliCity-0.yaml` for ImpliCity-0
* `./config/train_test/ImpliCity-mono.yaml` for ImpliCity-mono
* `./config/train_test/ImpliCity-stereo.yaml` for ImpliCity-stereo
* `./config/train_test/train_base.yaml` for default settings

The configuration files are hierarchically inherited with new values or additional fields. `ImpliCity-0.yaml` and `ImpliCity-mono.yaml` are inherited from `train_base.yaml`,
and `ImpliCity-stereo.yaml` is inherited from `ImpliCity-mono.yaml`.

    
### Evaluation
To evaluate a trained ImpliCity model, run:

```bash
python test.py <path/to/config.yaml>
```

Make sure to include the path to the trained model weights in the configuration file `config.yaml`:

```yaml
test:
   check_point: <path/to/model/weights.pt>
```


## Demo
### Download demo
We provide a demo for a quick run-through:
```
./data/ImpliCity_demo
├── data                        
│   ├── chunk_000               # input point cloud and query points
│   │   ├── ...
│   │   └── vis                 # visualization of the demo training data
│   │       └── ...
│   └── raster                  # tiff files including cropped orthorectified images, masks, ground truth DSM
│       └── ...
│   
├── expected_output             # expected output of different models (configurations) 
│   └── ...
└── model                       # pretrained models
    └── ...
```

Please use this script to download our demo:
```bash
bash ./scripts/download_demo.sh
```

Unfortunately, we cannot share our complete dataset due to the commercial nature of VHR imagery. In this demo, we thus provide a small
preprocessed demo dataset with a spatial extent of 64&times;64 m in world coordinates:

```yaml
463948.875, 5249174.125; 464012.875, 5249238.125 (EPSG:32632 - WGS 84 / UTM zone 32N - Projected)
```


### Run demo

To run the pretrained model with demo data, please run:
```bash
# ImpliCity-0 demo:
python test.py config/train_test/ImpliCity-0.yaml

# ImpliCity-mono demo:
python test.py config/train_test/ImpliCity-mono.yaml

# ImpliCity-stereo demo:
python test.py config/train_test/ImpliCity-stereo.yaml
```

To continue training, please run:
```bash
# ImpliCity-0 demo:
python train.py config/train_test/ImpliCity-0.yaml  --no-wandb

# ImpliCity-mono demo:
python train.py config/train_test/ImpliCity-mono.yaml  --no-wandb

# ImpliCity-stereo demo:
python train.py config/train_test/ImpliCity-stereo.yaml  --no-wandb
```


## Repository structure
```
.
├── config                 
│   ├── dataset            # configuration files for building the whole dataset
│   │   └── ...
│   └── train_test         # configuration files for training and inference 
│       └── ...             
├── data                   # data and pretrained models
├── out                    # output directory 
├── scripts                
│   ├── dataset_ImpliCity/build_dataset.py   # python script to build the whole dataset (only for reference, would not work without all source data)
│   └── download_demo.sh   # script to download demo data and pretrained models
├── src                    
│   └── ...                # source code of core modules
├── LICENSE
├── README.md              # this file
├── requirements.txt       # dependency list
├── test.py                # python script to test ImpliCity
└── train.py               # python script to train ImpliCity
```




## Contact
If you run into any problems or have questions, please contact [Bingxin Ke](mailto:bingke@ethz.ch) and [Corinne Stucker](mailto:corinne.stucker@geod.baug.ethz.ch).


## Citation

If you find our code or work useful, please cite:

```bibtex
@article{stucker2022implicity,
  title={ImpliCity: City Modeling from Satellite Images with Deep Implicit Occupancy Fields},
  author={Stucker, Corinne and Ke, Bingxin and Yue, Yuanwen and Huang, Shengyu and Armeni, Iro and Schindler, Konrad},
  journal={arXiv preprint arXiv:2201.09968},
  year={2022}
}
```
