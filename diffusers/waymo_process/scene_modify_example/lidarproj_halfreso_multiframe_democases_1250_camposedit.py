import os
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import math
import itertools
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.utils import box_utils
# from waymo_open_dataset.utils import keypoint_data
import pandas as pd
import multiprocessing
import warnings
from waymo_open_dataset.utils import transform_utils
import copy
warnings.filterwarnings("ignore")

def projpc(data,frame_points,intid2inboxpoints,waymoid2intid,idx,output_path,delta_x,delta_y,delta_theta,resomult=0.5,multiframe_num=2):
    
    if (not idx<multiframe_num) and (not idx>len(frame_points)-1-multiframe_num):
        merged_lidar_points = np.concatenate(frame_points[idx-multiframe_num:idx+multiframe_num+1],axis=0)
    elif idx<multiframe_num:
        merged_lidar_points = np.concatenate(frame_points[0:2*multiframe_num+1],axis=0)
    elif idx>len(frame_points)-1-multiframe_num:
        merged_lidar_points = np.concatenate(frame_points[len(frame_points)-2*multiframe_num-1:-1],axis=0)

    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    segment_name = frame.context.name
    
    objs_points=[]
    objs_depths=[]

    for label in frame.laser_labels:
        intid=waymoid2intid[label.id]
        if not intid in intid2inboxpoints:
            continue
        box = box_utils.box_to_tensor(label.box)[tf.newaxis, :]

        center = box[:, 0:3]
        dim = box[:, 3:6]
        heading = box[:, 6]
        # [M, 3, 3]
        rotation = transform_utils.get_yaw_rotation(heading)
        # [M, 4, 4]
        transform = transform_utils.get_transform(rotation, center)
        # [M, 4, 4]
        transform = tf.linalg.inv(transform)
        # [M, 3, 3]
        rotation = transform[:, 0:3, 0:3]
        # [M, 3]
        translation = transform[:, 0:3, 3]

        pointsinbox=[]

        if (not idx<multiframe_num) and (not idx>len(frame_points)-1-multiframe_num):
            lrange=idx-multiframe_num
            rrange=idx+multiframe_num+1
        elif idx<multiframe_num:
            lrange=0
            rrange=multiframe_num+1
        elif idx>len(frame_points)-1-multiframe_num:
            lrange=len(frame_points)-2*multiframe_num-1
            rrange=len(frame_points)

        for frame_ind in range(lrange,rrange):
            if frame_ind in intid2inboxpoints[intid]:
                pointsinbox.append(intid2inboxpoints[intid][frame_ind])

        if len(pointsinbox)==0:
            continue
        pointsinbox = np.concatenate(pointsinbox,axis=0)
        if len(pointsinbox)==0:
            continue

        
        pointsinbox_coord = pointsinbox[:,:3]-translation
        pointsinbox_coord = tf.einsum('nj,mij->nmi', pointsinbox_coord, np.linalg.inv(rotation))

        pointsinbox_coord=pointsinbox_coord.numpy().squeeze(1)


        mean_depth=np.linalg.norm(pointsinbox_coord[:,:2],axis=1).min()

        objs_depths.append(mean_depth)

        vehicle_to_world = np.array([
            frame.pose.transform
        ]).reshape(4,4)

        pointsinbox_coord = np.dot(vehicle_to_world[:3, :3], pointsinbox_coord.T) + vehicle_to_world[:3, 3].reshape(-1, 1)
        pointsinbox_coord = pointsinbox_coord.T


        pointsinbox = np.concatenate([pointsinbox_coord,pointsinbox[:,3:]],axis=1)

        objs_points.append(pointsinbox)
        # merged_lidar_points = np.concatenate([merged_lidar_points,pointsinbox],axis=0)

    imgname=[]
    imgname2uv_filtmedian=[]
    imgname2colors_filtmedian=[]

    objs_points = [a for a, _ in sorted(zip(objs_points, objs_depths), key=lambda x: x[1])]

    for i, image in enumerate(frame.images):
        camera_calibration = next(cc for cc in frame.context.camera_calibrations if cc.name == image.name)
        frame_num = idx
        camera_position = open_dataset.CameraName.Name.Name(image.name)
        cc=camera_calibration

        if not camera_position=='FRONT':
            continue

        image_filename = f"{segment_name}_{str(idx)}_{camera_position}.jpg"
        

        origin_extrinsic = cc.extrinsic.transform#ind 3，7，11=x,y,z
        extrinsic = copy.deepcopy(origin_extrinsic)
        extrinsic[3]+=delta_x
        extrinsic[7]+=delta_y

        extrinsic = np.reshape(extrinsic, [4, 4])

        beta=np.radians(delta_theta)
        Rz_4x4 = np.array([  
            [np.cos(beta), -np.sin(beta), 0, 0],  
            [np.sin(beta), np.cos(beta), 0, 0],  
            [0,           0,            1, 0],  
            [0,           0,            0, 1]  
        ])  
        extrinsic = np.dot(Rz_4x4, extrinsic)  

        extrinsic=tf.convert_to_tensor(extrinsic,dtype='float32')

        intrinsic = cc.intrinsic
        intrinsic[0]*=resomult
        intrinsic[1]*=resomult
        intrinsic[2]*=resomult
        intrinsic[3]*=resomult
        intrinsic[6]*=resomult
        intrinsic[7]*=resomult
        intrinsic = tf.constant(cc.intrinsic)

        if 'FRONT' in camera_position:
            img_size=[960,640]#[1920,1280]#800,533#0.41666666,0.416406
        else:
            img_size=[960,443]#[1920, 886]#800,369

        metadata = tf.constant([img_size[0], img_size[1], cc.rolling_shutter_direction])

        img=image
        camera_image_metadata = tf.constant([
            *img.pose.transform,
            img.velocity.v_x, img.velocity.v_y, img.velocity.v_z,
            img.velocity.w_x, img.velocity.w_y, img.velocity.w_z,
            img.pose_timestamp,
            img.shutter,
            img.camera_trigger_time,
            img.camera_readout_done_time
        ], dtype=tf.float32)


        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        mask = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8)
        for objpoints in objs_points:
            proj_all = py_camera_model_ops.world_to_image(
                extrinsic, intrinsic, metadata, camera_image_metadata, objpoints[:,:3]
            )
            u=proj_all.numpy()[:,0].round()
            v=proj_all.numpy()[:,1].round()

            valid_points = np.where((u >= 0) & (u < img_size[0]) & (v >= 0) & (v < img_size[1]))

            u = u[valid_points].astype(int)
            v = v[valid_points].astype(int)


            dists=objpoints[valid_points,:3]-np.array(image.pose.transform).reshape(4,4)[:3,3]
            dists=tf.norm(dists,axis=-1).numpy()


            uv_filtered, dists_filtered,colors=filter_duplicates(np.stack([v,u]).T,dists.T,objpoints[valid_points,::-1][0,:,:3])
            tempimg = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)    
            tempimg[uv_filtered[:,0].astype(int),uv_filtered[:,1].astype(int)] = colors
            img += tempimg*mask

            if len(uv_filtered)>0:
                proj_territory_size=4

                added_uvs=[]
                for radius in range(0,proj_territory_size+1):
                    added_uvs.append(uv_filtered+np.array([proj_territory_size,0]))
                    added_uvs.append(uv_filtered+np.array([-proj_territory_size,0]))
                    added_uvs.append(uv_filtered+np.array([0,-proj_territory_size]))
                    added_uvs.append(uv_filtered+np.array([0,proj_territory_size]))
                added_uvs=np.concatenate(added_uvs,axis=0)
                added_uvs=np.concatenate([uv_filtered,added_uvs],axis=0)

                valid_points = np.where((added_uvs[:,0] >= 0) & (added_uvs[:,0] < img_size[1]) & (added_uvs[:,1] >= 0) & (added_uvs[:,1] < img_size[0]))
                added_uvs=added_uvs[valid_points]
                mask[added_uvs[:,0].astype(int),added_uvs[:,1].astype(int)]=0

        proj_all = py_camera_model_ops.world_to_image(
            extrinsic, intrinsic, metadata, camera_image_metadata, merged_lidar_points[:,:3]
        )

        u=proj_all.numpy()[:,0].round()
        v=proj_all.numpy()[:,1].round()

        valid_points = np.where((u >= 0) & (u < img_size[0]) & (v >= 0) & (v < img_size[1]))

        u = u[valid_points].astype(int)
        v = v[valid_points].astype(int)

        dists=merged_lidar_points[valid_points,:3]-np.array(image.pose.transform).reshape(4,4)[:3,3]
        dists=tf.norm(dists,axis=-1).numpy()

        uv_filtered, dists_filtered,colors=filter_duplicates(np.stack([v,u]).T,dists.T,merged_lidar_points[valid_points,::-1][0,:,:3])
        # Draw Lidar points on the image
        # img[v, u] = [255, 255, 255]
        imgname.append(image_filename)

        # Create empty image
        tempimg = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)    
        tempimg[uv_filtered[:,0].astype(int),uv_filtered[:,1].astype(int)] = colors
        img += tempimg*mask
    

        output_filename=output_path+image_filename

        #取靠下60%的img
        if 'FRONT' in camera_position:
            img=img[256:]
        else:
            img=img[59:]
        
        cv2.imwrite(output_filename,img)

        
# import torch
def save_images_from_waymo(segment_path, output_path):

    waymoid2intid={}
    intid2inboxpoints={}

    count=1
    # 读取Waymo数据集
    dataset = tf.data.TFRecordDataset(segment_path, compression_type='')
    camera_poses={}
    frame_points=[]
    segname=segment_path.split('segment-')[1].split('_with')[0]

    print(segname)
    
    
    for idx, data in enumerate(dataset):

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        segment_name = frame.context.name

        range_images, camera_projections,seg_labels,range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        points_all = np.concatenate([points_all,np.zeros([points_all.shape[0],1])], axis=1)
        # camera projection corresponding to each point.
        cp_points_all = np.concatenate(cp_points, axis=0)

        cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        images = sorted(frame.images, key=lambda i:i.name)

        points_withcolor=[]

        for image_ind,image in enumerate(images):

            mask = tf.equal(cp_points_all_tensor[..., 0], images[image_ind].name)

            cp_points_all_tensor_inimg = tf.cast(tf.gather_nd(
                cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)

            points_all_inimg = tf.gather_nd(points_all, tf.where(mask))

            projected_points_all_from_raw_data = cp_points_all_tensor_inimg[..., 1:3].numpy()

            projected_coords=projected_points_all_from_raw_data[:,:2].astype('int')

            projected_coords=projected_coords[:, ::-1]

            imagetensor=tf.image.decode_jpeg(image.image)

            pointcolors=tf.gather_nd(imagetensor, projected_coords)

            points_all_inimg=np.concatenate([points_all_inimg.numpy(),pointcolors.numpy()],axis=1)

            points_withcolor.append(points_all_inimg)


        points_withcolor = np.concatenate(points_withcolor,axis=0)

        for label in frame.laser_labels:
            if not label.id in waymoid2intid:
                waymoid2intid[label.id]=len(waymoid2intid)+1
            box = box_utils.box_to_tensor(label.box)[tf.newaxis, :]
            #box: [M, 7] tensor. Inner dims are: [center_x, center_y, center_z, length,width, height, heading].
            inbox_inds=box_utils.is_within_box_3d(points_withcolor[:,:3], box)[:, 0]
            points_withcolor[inbox_inds,3] = waymoid2intid[label.id]

            center = box[:, 0:3]
            dim = box[:, 3:6]
            heading = box[:, 6]
            # [M, 3, 3]
            rotation = transform_utils.get_yaw_rotation(heading)
            # [M, 4, 4]
            transform = transform_utils.get_transform(rotation, center)
            # [M, 4, 4]
            transform = tf.linalg.inv(transform)
            # [M, 3, 3]
            rotation = transform[:, 0:3, 0:3]
            # [M, 3]
            translation = transform[:, 0:3, 3]

            # [N, M, 3]
            point_in_box_frame = tf.einsum('nj,mij->nmi', points_withcolor[inbox_inds,:3], rotation) + translation


            point_in_box_frame = point_in_box_frame.numpy().squeeze(1)

            point_in_box_frame = np.concatenate([point_in_box_frame,points_withcolor[inbox_inds][:,4:]],axis=1)

            # if len(point_in_box_frame)>0:
            if not waymoid2intid[label.id] in intid2inboxpoints:
                intid2inboxpoints[waymoid2intid[label.id]]={}
                intid2inboxpoints[waymoid2intid[label.id]][idx]=point_in_box_frame
            else:
                intid2inboxpoints[waymoid2intid[label.id]][idx]=point_in_box_frame

        vehicle_to_world = np.array([
            frame.pose.transform
        ]).reshape(4,4)

        transformed_lidar_points = np.dot(vehicle_to_world[:3, :3], points_withcolor[:,:3].T) + vehicle_to_world[:3, 3].reshape(-1, 1)
        transformed_lidar_points = transformed_lidar_points.T

        temp_ind=np.where(points_withcolor[:,3]==0)

        transformed_lidar_points=np.concatenate([transformed_lidar_points[temp_ind],points_withcolor[temp_ind][:,4:]],axis=1)

        frame_points.append(transformed_lidar_points)


    
    output_path=output_path+segname+'/'
    os.makedirs(output_path,exist_ok=True)

    start_frame=0

    unit_frames=25
    y_max=4.0/unit_frames
    theta_max=35 / (2*unit_frames)
    delta_xs=[x*y_max for x in range(0,unit_frames)]+[(unit_frames-x)*y_max for x in range(0,unit_frames)] + \
                [0 for i in range(unit_frames)] + [0 for i in range(unit_frames)]  + \
                [0 for i in range(unit_frames)] + [0 for i in range(unit_frames)]  + \
                [x*y_max for x in range(0,unit_frames)]+[x*y_max+24*y_max for x in range(0,unit_frames)]

    delta_ys=   [0 for i in range(unit_frames)] + [0 for i in range(unit_frames)] +\
                [x*y_max for x in range(0,unit_frames)]+[(unit_frames-x)*y_max for x in range(0,unit_frames)] +\
                [-x*y_max for x in range(0,unit_frames)]+[-(unit_frames-x)*y_max for x in range(0,unit_frames)] +\
                [0 for i in range(unit_frames)] + [0 for i in range(unit_frames)]
    
    delta_thetas=[0 for i in range(len(delta_ys))]

    for idx, data in enumerate(dataset):

        if idx>len(frame_points):
            break

        if idx%2==1:#skip 1/2 frames
            continue

        projpc(data,frame_points,intid2inboxpoints,waymoid2intid,idx,output_path,delta_x = delta_xs[idx+start_frame], delta_y=delta_ys[idx+start_frame],delta_theta=delta_thetas[idx+start_frame])

        

def filter_duplicates(uv, dists,colors):

    data = np.column_stack((uv, dists,colors))
    df = pd.DataFrame(data, columns=['u', 'v', 'dist','r','g','b'])

    df_sorted = df.sort_values(by='dist', ascending=True).groupby(['u', 'v'], as_index=False).first()

    uv_filtered = df_sorted[['u', 'v']].values
    dists_filtered = df_sorted['dist'].values.reshape(-1, 1)
    colors=df_sorted[['r','g','b']].values

    return uv_filtered, dists_filtered,colors

import sys
if __name__ == "__main__":

    folder_path='/mnt/nvme1n1p1/waymo/training_200segs'

    output_dir = "./waymo_pseudoimg_multiframe_5hz_democases/"

    os.makedirs(output_dir,exist_ok=True)

    valscenes=sorted(os.listdir(folder_path))

    scenes= [
    '12505030131868863688_1740_000_1760_000',
    ]

    valscenes = ['segment-'+temp+'_with_camera_labels.tfrecord' for temp in scenes]


    for idx,tfrecord_path in enumerate(valscenes):
    #   if idx%200==fileidx:
        
        file_path = os.path.join(folder_path, tfrecord_path)

        # 设置输出路径

        # 运行保存相机成像的函数
        save_images_from_waymo(file_path, output_dir)
