# FreeVS
Official implementation of [ICLR2025] FreeVS: Generative View Synthesis on Free Driving Trajectory.

[Qitai Wang](https://github.com/esdolo), [Lue Fan](https://lue.fan/), [Yuqi Wang](https://robertwyq.github.io/), [Yuntao Chen](https://scholar.google.com/citations?user=iLOoUqIAAAAJ), [Zhaoxiang Zhang](https://zhaoxiangzhang.net/)


[[arXiv](https://arxiv.org/abs/2410.18079 )] [[Project page](https://freevs24.github.io//)]

[![Demo video](demos/12505030131868863688_1740_000_1760_000_FRONT.mp4)]

## Recent updates
- **[2025/02/08]** Implementation of FreeVS on Waymo Open Dataset is released.
- **[2025/01/23]** 🎉 FreeVS was accepted to ICLR 2025！


## Prerequisite
```bash
conda create -n freevs python=3.8
conda activate freevs

cd diffusers
pip install .
pip install -r requirements.txt
```

## To Do
- [ ] Implementation on nuScenes

# Waymo Open Dataset  

## Prepare Waymo GT images / pseudo images
```bash
cd waymo_process

# Extract images from .tfrecord files
python extract_gt_images.py --waymo_raw_dir <path to WOD> --output_dir waymo_gtimg_5hz_allseg --interval 2

# Generating pseudo-image
python lidarproj_halfreso_multiframe.py --waymo_raw_dir <path to WOD> --output_dir waymo_pseudoimg_multiframe --interval 2 

# Generating pseudo-image is time-consuming. You can also use the multiprocess script:
bash gen_pseudo_img.bash

cd ..

# Generate pickle info file
python data_process/waymo_data_generation_subsegbycampos_multiframe.py --data_root waymo_process/waymo_gtimg_5hz_allseg/ --pseudoimg_root waymo_process/waymo_pseudoimg_multiframe/ --output_pickle waymo_process/waymo_multiframe_subsegbycampos.pkl
```

## (Recommend) Additional pseudo-images for camera transformation simulation 
```bash
cd waymo_process

python lidarproj_halfreso_multiframe_mismatchframeaug.py --waymo_raw_dir <path to WOD> --output_dir waymo_pseudoimg_multiframe_+4frame --interval 2 --mismatchnframe 4

python lidarproj_halfreso_multiframe_mismatchframeaug.py --waymo_raw_dir <path to WOD> --output_dir waymo_pseudoimg_multiframe_-4frame --interval 2 --mismatchnframe -4

cd ..

python data_process/waymo_data_generation_subsegbycampos_multiframe.py --data_root waymo_process/waymo_gtimg_5hz_allseg/ --pseudoimg_root waymo_process/waymo_pseudoimg_multiframe/ --transformation_simulation --pseudoimg_root_2 waymo_process/waymo_pseudoimg_multiframe_+4frame/ --pseudoimg_root_3 waymo_process/waymo_pseudoimg_multiframe_+4frame/ --output_pickle waymo_process/waymo_multiframe_subsegbycampos_transform_simulation.pkl
```

## Train SVD
We initialize SVD model from https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt. Download it as pretrained/stable-video-diffusion-img2vid-xt.
```bash
# On WOD, we recommend training diffuser model with a frozen pseudo-image encoder, which can significantly accelerate model convergence.
# We privide a pseudo-image encoder checkpoint in diffusers/pretrained/.
bash examples/freevs/scripts/run_train_onlyunet.sh

# Script for joint training pseudo-img encoder and diffuser
bash examples/freevs/scripts/run_train.sh
```

## Run Inference
```bash
python examples/freevs/inference_svd.py --model_path work_dirs/freevs_waymo_halfreso_multiframe --img_pickle waymo_process/waymo_multiframe_subsegbycampos.pkl --output_dir rendered_waymo_origin
```
To control the camera pose for novel trajectory simulation, please modify camera extrinsic in waymo_process/lidarproj_halfreso_multiframe.py. We provide a example case of camera pose editing in scene_modify_example/lidarproj_halfreso_multiframe_democases_1250_camposedit.py.

## Citation
```
@article{wang2024freevs,
  title={Freevs: Generative view synthesis on free driving trajectory},
  author={Wang, Qitai and Fan, Lue and Wang, Yuqi and Chen, Yuntao and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2410.18079},
  year={2024}
}
```

## Acknowledgement 
Many thanks to the following open-source projects:
* [diffusers](https://github.com/huggingface/diffusers)