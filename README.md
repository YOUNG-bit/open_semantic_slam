<h1 align="center"> OpenGS-SLAM: Open-Set Dense Semantic SLAM with 3D Gaussian Splatting for Object-Level Scene Understanding </h1>

<h3 align="center"> Dianyi Yang, Yu Gao, Xihan Wang, Yufeng Yue, Yi Yang∗, Mengyin Fu </h3>

<!-- <h3 align="center">
  <a href="https://arxiv.org/abs/2408.12677">Paper</a> | <a href="https://youtu.be/rW8o_cRPZBg">Video</a> | <a href="https://gs-fusion.github.io/">Project Page</a>
</h3> -->

<h3 align="center">
  <a href="https://www.youtube.com/watch?v=uNJ4vTpfGU0">Video</a> | <a href="https://young-bit.github.io/opengs-github.github.io/">Project Page</a>
</h3>


<p align="center">
  <a href="">
    <img src="./media/github.gif" alt="teaser" width="100%">
  </a>
</p>

<p align="center"> All the reported results are obtained from a single Nvidia RTX 4090 GPU. </p>

Abstract: *Recent advancements in 3D Gaussian Splatting have significantly improved the efficiency and quality of dense semantic SLAM. However, previous methods are generally constrained by limited-category pre-trained classifiers and implicit semantic representation, which hinder their performance in open-set scenarios and restrict 3D object-level scene understanding. To address these issues, we propose OpenGSSLAM, an innovative framework that utilizes 3D Gaussian representation to perform dense semantic SLAM in open-set environments. Our system integrates explicit semantic labels derived from 2D foundational models into the 3D Gaussian framework, facilitating robust 3D object-level scene understanding. We introduce Gaussian Voting Splatting to enable fast 2D label map rendering and scene updating. Additionally, we propose a Confidence-based 2D Label Consensus method to ensure consistent labeling across multiple views. Furthermore, we employ a Segmentation Counter Pruning strategy to improve the accuracy of semantic scene representation. Extensive experiments on both synthetic and real-world datasets demonstrate the effectiveness of our method in scene understanding, tracking, and mapping, achieving 10× faster semantic rendering and 2× lower storage costs compared to existing methods.*



## Environments
Install requirements
```bash
conda create -n opengsslam python==3.9
conda activate opengsslam
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
Install submodules

```bash
conda activate opengsslam
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Scene Interaction Demo
### 1. Download our pre-constructed Semantic 3D Gaussian scenes for the Replica dataset from the following link: [Driver](https://drive.google.com/drive/folders/1-bGoaZQRRKLHXFQGq3_6gu1KXhoePbQv?usp=drive_link) 

### 2. Scene Interaction
```
python ./final_vis.py --scene_npz [download_path]/room1.npz
```
Here, users can click on any object in the scene to interact with it and use our Gaussian Voting method for real-time semantic rendering. Note that we use the **pynput** library to capture mouse clicks, which retrieves the click position on **the entire screen**. To map this position to the display window, we subtract an offset `(x_off, y_off)`, representing the window’s top-left corner on the screen. All tests were conducted on an Ubuntu system with a 2K resolution.

### *Key Press Description*

- **T**: Toggle between color and label display modes.  
- **J**: Toggle between showing all objects or a single object.  
- **K**: Capture the current view.  
- **A**: Translate the object along the x-axis by +0.01.  
- **S**: Translate the object along the y-axis by +0.01.  
- **D**: Translate the object along the z-axis by +0.01.  
- **Z**: Translate the object along the x-axis by -0.01.  
- **X**: Translate the object along the y-axis by -0.01.  
- **C**: Translate the object along the z-axis by -0.01.  
- **F**: Rotate the object around the x-axis by +1 degree.  
- **G**: Rotate the object around the y-axis by +1 degree.  
- **H**: Rotate the object around the z-axis by +1 degree.  
- **V**: Rotate the object around the x-axis by -1 degree.  
- **B**: Rotate the object around the y-axis by -1 degree.  
- **N**: Rotate the object around the z-axis by -1 degree.  
- **O**: Output the current camera view matrix.  
- **M**: Switch to the next mapping camera view.  
- **L**: Increase the scale of all Gaussians.  
- **P**: Downsample Gaussians using a voxel grid.  


## SLAM Source Code

Coming soon!

<!-- ## Note

This repository contains the code used in the paper "OpenGS-SLAM: Open-Set Dense Semantic SLAM with 3D Gaussian Splatting for Object-Level Scene Understanding". The full code will be released upon acceptance of the paper. -->

## Acknowledgement
We sincerely thank the developers and contributors of the many open-source projects that our code is built upon.

* [GS_ICP_SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM)
* [SplaTAM](https://github.com/spla-tam/SplaTAM/tree/main)

## Citation

If you find our paper and code useful, please cite us:
```bibtex
@article{yang2025opengs,
  title={OpenGS-SLAM: Open-Set Dense Semantic SLAM with 3D Gaussian Splatting for Object-Level Scene Understanding},
  author={Yang, Dianyi and Gao, Yu and Wang, Xihan and Yue, Yufeng and Yang, Yi and Fu, Mengyin},
  journal={arXiv preprint arXiv:2503.01646},
  year={2025}
}
```
