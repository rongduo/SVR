## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate SVR
```

Compile and Install the rasterizer with frequency constraints and depth rendering:
```bash
pip install submodules/diff-gaussian-rasterization-freqConstrain
pip install submodules/simple-knn  # also install the simple-knn library
```

## Data Preparation
``` 
cd SVR
mkdir dataset 
cd dataset

# download LLFF dataset
Please download the LLFF dataset from the link: https://drive.google.com/drive/folders/1QhpVUIjp9kwuCXTMqUC3jMYZFY4QPcHU and put the scenes within a subfolder named 'nerf_llff_data' in dataset folder.

# run colmap to obtain initial point clouds with limited viewpoints
python tools/colmap_llff.py
``` 

## Training
Train the method on LLFF dataset with 3 views
``` 
python train.py  --source_path dataset/nerf_llff_data/horns --model_path output/horns --eval  --n_views 3 --sample_pseudo_interval 1
``` 
## Results 
In folder 'results/fern', we provide the reconstructed point cloud and tensorboard record for the scene 'fern' from LLFF dataset.

## Rendering
Run the following script to render the images.  

```
python render.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000
```

You can customize the rendering path as same as NeRF by adding `video` argument

```
python render.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000  --video  --fps 30
```

## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000
```

