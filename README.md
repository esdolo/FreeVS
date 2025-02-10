# FreeVS: Generative View Synthesis on Free Driving Trajectory
Official implementation of **[ICLR2025]** FreeVS: Generative View Synthesis on Free Driving Trajectory.

[Qitai Wang](https://github.com/esdolo), [Lue Fan](https://lue.fan/), [Yuqi Wang](https://robertwyq.github.io/), [Yuntao Chen](https://scholar.google.com/citations?user=iLOoUqIAAAAJ), [Zhaoxiang Zhang](https://zhaoxiangzhang.net/)


[[arXiv](https://arxiv.org/abs/2410.18079 )] [[Project page](https://freevs24.github.io//)]

![Demo video](diffusers/demos/12505030131868863688_1740_000_1760_000_FRONT.gif)

## Recent updates
- **[2025/02/10]** Implementation of FreeVS on Waymo Open Dataset is released.
- **[2025/01/23]** 🎉 FreeVS was accepted to ICLR 2025！

## To Do
- [ ] Implementation on nuScenes
- [ ] Provide 3D prior based on estimated depth where LiDAR observations are missing, to ensure the consistency of far, background area.

## Prerequisite
```bash
conda create -n freevs python=3.8
conda activate freevs

cd diffusers
pip install .
pip install -r requirements.txt
```

# Waymo Open Dataset  

## Quick Start with Examples
Download a trained model [checkpoint](https://huggingface.co/Esdolo/FreeVS_WOD), as well as serveral processed example scenes. **Please check the [License Agreement](https://waymo.com/open/terms/) of WOD dataset before downloading this checkpoint**.
```bash
cd diffusers
pip install huggingface_hub

huggingface-cli download Esdolo/FreeVS_WOD --local-dir ./pretrained/FreeVS_WOD/

huggingface-cli download Esdolo/FreeVS_Examples --local-dir ./waymo_process/FreeVS_Examples/

cd waymo_process/FreeVS_Examples
tar -xzf FreeVS_Examples.tar.gz
cd ../..
```

Run inference with example scenes:
```bash
python examples/freevs/inference_svd.py --front_only --model_path pretrained/FreeVS_WOD/ --img_pickle waymo_process/FreeVS_Examples/waymo_example_newtraj.pkl  --output_dir rendered_waymo_example_newtraj

python examples/freevs/inference_svd.py --front_only --model_path pretrained/FreeVS_WOD/ --img_pickle waymo_process/FreeVS_Examples/waymo_example_origintraj.pkl  --output_dir rendered_waymo_example_origintraj 
```
Results synthesized in the origin/new trajectory will be output to rendered_waymo_example_origintraj / rendered_waymo_example_newtraj.

## Prepare Waymo GT images / pseudo images
```bash
cd waymo_process

#|-- <path to WOD>
#     |--*.tfrecoed
#     |...

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
python examples/freevs/inference_svd.py --model_path work_dirs/freevs_waymo_halfreso_multiframe_transformation_simulate_trainunet --img_pickle waymo_process/waymo_multiframe_subsegbycampos_transform_simulation.pkl --output_dir rendered_waymo_origin 
```
To control the camera pose for novel trajectory simulation, please modify camera extrinsic in waymo_process/lidarproj_halfreso_multiframe.py. We provide a example case of camera pose editing in waymo_process/scene_modify_example/lidarproj_halfreso_multiframe_democases_1250_camposedit.py.

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
* [Drive-WM](https://github.com/BraveGroup/Drive-WM)
