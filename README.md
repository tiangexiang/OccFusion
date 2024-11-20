<div align="center">

<h1>OccFusion: Rendering Occluded Humans with Generative Diffusion Priors</h1>

<div>
    <a href="https://adamsunn.github.io/" target="_blank">Adam Sun*</a>;
    <a href="https://ai.stanford.edu/~xtiange/" target="_blank">Tiange Xiang*</a>;
    <a href="https://profiles.stanford.edu/scott-delp" target="_blank">Scott Delp</a>;
    <a href="https://profiles.stanford.edu/fei-fei-li" target="_blank">Li Fei-Fei^</a>;
    <a href="https://profiles.stanford.edu/ehsan-adeli" target="_blank">Ehsan Adeli^</a>;
</div>
<div>
    <em> * Equal contributions; Junior author listed first. </em>
    <br>
    <em>^ Equal mentorship.  </em>
    <br>
    Stanford University
</div>
<div>
    NeurIPS 2024
</div>

<div style="width: 95%; text-align: center; margin:auto;">
    <img style="width:100%" src="./teaser.png"><br>
    <em>OccFusion recovers occluded human from monocular videos with <strong>only 10mins of training.</strong></em>
</div>

For more visual results, go checkout our <a href="https://cs.stanford.edu/~xtiange/projects/occfusion/" target="_blank">project page</a>.  
For details, please refer to our <a href="https://arxiv.org/pdf/2407.00316" target="_blank">paper</a>.

<div align="left">

## Environment
To start with, please clone our envrionment:

```bash
    conda env create -f environment.yml
    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn
```
## Data and Necessary Assets

### 1. OcMotion sequences
We provide training/rendering code for the 6 <a href="https://github.com/boycehbz/CHOMP">OcMotion</a> sequences that are sampled by <a href="https://github.com/tiangexiang/Wild2Avatar">Wild2Avatar</a>. If you find the preprocessed sequences usefull, please consider to cite [Wild2Avatar](https://arxiv.org/pdf/2401.00431.pdf) and [CHOMP](https://arxiv.org/pdf/2207.05375.pdf).    

Please download the processed sequences <a href="https://drive.google.com/drive/folders/1w9FzyKOhxQdhr_nmANRfRkXNdt9_M6BC?usp=sharing">here</a> and unzip the downloaded sequences in the `./data/` directory. The structure of `./data/` should look like:
```
./
├── ...
└── data/
    ├── 0011_02_1_w2a/
        ├── images/
        ├── masks/
        └── ...
    ├── 0011_02_2_w2a/
    ├── 0013_02_w2a/
    ├── 0038_04_w2a/
    ├── 0039_02_w2a/
    └── 0041_00_w2a/
```

### 2. SMPL model
Please register and download the <strong>neutral SMPL model</strong> [here](https://smplify.is.tue.mpg.de/download.php). Put the downloaded models in the folder `./assets/`. 

### 3. Canonical OpenPose canvas
To enable more efficient canonical space SDS, OpenPose canvas for canonical 2D poses are precomputed and can be downloaded [here](https://drive.google.com/drive/folders/1ITm_GB7LY5igY-80p6SCTy1LKBjv8M3Y?usp=sharing). Put the downloaded folder in the folder: `./assets/`. 

### (optional) 4. SAM-HQ weights
For training the model in Stage 0 (optional, see below), we need to compute binary masks for complete human inpaintings. We utilized [SAM-HQ](https://github.com/SysCV/sam-hq) for segmentation. If you wish to compute the masks by your own, please download the pretrained weights `sam_hq_vit_h.pth` [here](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view) and put the donwloaded weights in the folder: `./assets/`. 


After successful downloading, the structure of `./assets/` should look like
```
./
├── ...
└── assets/
    ├── daposesv2
        ├── -5.png
        └── ...
    ├── SMPL_NEUTRAL.pkl
    └── sam_hq_vit_h.pth (optional)
```

## Pretrained Models

We provide our pretrained models for all the OcMotion sequences to allow quick inference/evaluations. Please download the `ocmotion/` folder [here](https://drive.google.com/file/d/1SrEIZZJplAMtRJ4AfhBdYcUO1UDiI1jK/view?usp=sharing) and put the downloaded folders in `./output/`. 

## Usage

The training of OccFusion consists of 4 sequential stages. Stage 0 and 2 are optional that inpaint occluded human with customized models, sovlers, and prompts. Different combinations may impact the inpainting results greatly. *A high-quality pose conditioned human genertaion is out of the scope of thie work.* We provide our code (see Stage 0 and Stage 2 below) to allow users to try themselves. 

We provide our precomputed generations (to replicate our results in the paper) to be downloaded [here](https://drive.google.com/file/d/158wWKXhHWk-p9Y-hUTkB6YgZ2Wg1K1Oe/view?usp=sharing). Please unzip and put the `oc_generations/` folder directly on the root directory. Stage 0 and 2 can be skipped if use our computations. 

### (optional) Stage 0 

Run Stage 0 to segment binary masks for complete humans to be generated from stable diffusion models. To run Stage 0 on a OcMotion sequence, uncomment the corresponding `SUBJECT` varaible and
```bash
source run_oc_stage0.sh
``` 

The segmented binary maskes will be saved in the `./oc_genertaions/$SUBJECT/gen_masks/` directory.

### Stage 1

With the binary masks, run Stage 1 to start *Optimization*. To run Stage 1 on a OcMotion sequence, uncomment the corresponding `SUBJECT` varaible and
```bash
source run_oc_stage1.sh
``` 

The checkpoint along with renderings will be save in `./output/$SUBJECT/`.

### (optional) Stage 2

With an optimized model, run Stage 2 to launch incontext-inpainting. To run Stage 2 on a OcMotion sequence, uncomment the corresponding `SUBJECT` varaible and
```bash
source run_oc_stage2.sh
``` 

The inpainted RGB images will be saved in the `./oc_genertaions/$SUBJECT/incontext_inpainted/` directory.

### Stage 3

Lastly, with the inpainted RGB images and the optimized model checkpoint, run Stage 3 to start *Refinement*. To run Stage 3 on a OcMotion sequence, uncomment the corresponding `SUBJECT` varaible and
```bash
source run_oc_stage1.sh
``` 

The checkpoint along with renderings will be save in `./output/$SUBJECT/`.

### Rendering

At Stage 1 and 3, a rendering process will be trigered automatically after the training finishes. To explicitly render on a trained checkpoint, run
```bash
source render.sh
``` 

## Citation  
<!-- --- -->

If you find this repo useful in your work or research, please cite:

```bibtex
@inproceedings{occfusion,
  title={OccFusion: Rendering Occluded Humans with Generative Diffusion Priors},
  author={Sun, Adam and Xiang, Tiange and Delp, Scott and Fei-Fei, Li and Adeli, Ehsan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  url={https://arxiv.org/abs/2407.00316}, 
  year={2024}
}
```
