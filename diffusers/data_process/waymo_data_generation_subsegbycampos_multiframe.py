import os
import shutil
import cv2
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import json
import pickle as pkl
from ipdb import set_trace

def process_and_save_image(img_path, output_path):
    img = cv2.imread(img_path)

    resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    # clip sky
    if 'FRONT' in img_path.split('/')[-1]:
        resized_img=resized_img[256:]
    else:
        resized_img=resized_img[59:]

    cv2.imwrite(output_path, resized_img)

import argparse
parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--data_root', type=str,default='waymo_process/waymo_gtimg_5hz_allseg')
parser.add_argument('--pseudoimg_root', type=str,default = 'waymo_process/waymo_pseudoimg_multiframe')
parser.add_argument('--transformation_simulation', action='store_true', default=False)
parser.add_argument('--pseudoimg_root_2', type=str, default='waymo_process/waymo_pseudoimg_multiframe') 
parser.add_argument('--pseudoimg_root_3', type=str, default='waymo_process/waymo_pseudoimg_multiframe') 
parser.add_argument('--output_dir', type=str, default='waymo_process/gt') 
parser.add_argument('--output_pickle', type=str, default='waymo_process/waymo_multiframe_subsegbycampos.pkl') 
args = parser.parse_args()

if __name__ == "__main__":

    data_root = args.data_root
    pseudoimg_root = args.pseudoimg_root
    if args.transformation_simulation:
        pseudoimg_root_2 = args.pseudoimg_root_2
        pseudoimg_root_3 = args.pseudoimg_root_3
    else:
        pseudoimg_root_2 = args.pseudoimg_root
        pseudoimg_root_3 = args.pseudoimg_root
    output_dir = args.output_dir
    output_pickle = args.output_pickle

    os.makedirs(output_dir, exist_ok=True)

    scenes = sorted(os.listdir(data_root))

    scenes = [temp.split('segment-')[-1].split('_with_camera')[0] for temp in scenes]

    infos = defaultdict(list)

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_root, scene, '')
        out_scene_path = os.path.join(output_dir, scene)
        pseudo_scene_path = os.path.join(pseudoimg_root, scene)

        imgs_path = scene_path

        os.makedirs(out_scene_path, exist_ok=True)

        frames = sorted(os.listdir(imgs_path))

        for cam_pos in ['FRONT','FRONT_LEFT','SIDE_LEFT','FRONT_RIGHT','SIDE_RIGHT']:
            for frame in frames:
                if not cam_pos+'.jpg' in frame:
                    continue
                frame_path = os.path.join(imgs_path, frame)
                out_frame_path = os.path.join(out_scene_path, frame)
                pseudo_frame_path = os.path.join(pseudo_scene_path, frame[5:])
                pseudo_frame_path_2 = os.path.join(os.path.join(pseudoimg_root_2, scene), frame[5:])
                pseudo_frame_path_3 = os.path.join(os.path.join(pseudoimg_root_3, scene), frame[5:])

                if not os.path.exists(out_frame_path):
                    process_and_save_image(frame_path, out_frame_path)
                # import pdb
                # pdb.set_trace()

                frame_info = dict()
                frame_info['image'] = out_frame_path          
                frame_info['pseudo_image'] = pseudo_frame_path  
                frame_info['pseudo_image_minusframe'] = pseudo_frame_path_2  
                frame_info['pseudo_image_plusframe'] = pseudo_frame_path_3  
                infos[scene+'_'+cam_pos].append(frame_info)
        # print('finish scene {}'.format(scene))

    with open(output_pickle, 'wb') as fw:
        pkl.dump(infos, fw)
