#! /srv/rail-lab/flash5/mpatel377/anaconda3/envs/ovr/bin/python

import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image, ImageDraw
from datasets import load_dataset
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

dataset_procthor = prior.load_dataset("procthor-10k")

controller = Controller(scene=dataset_procthor["train"][0],
                        renderInstanceSegmentation=True,
                        platform=CloudRendering)

def get_top_down_frame():
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", 
                            raise_for_failure=True, 
                            platform=CloudRendering)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)

def get_obj_loc_and_ids(this_house):

    object_locations = {0:[],1:[]}
    object_ids = {0:[],1:[]}

    def append_pos(obj, depth=0):
        # print('   '*depth+obj['id'])
        object_ids[depth].append(obj['id'])
        if 'position' in obj:
            object_locations[depth].append(obj['position'])
        if 'children' in obj:
            for c in obj['children']:
                append_pos(c, depth+1)

    for obj in this_house['objects']:
        append_pos(obj)

    return object_locations, object_ids

def get_useful_pose_set(this_controller, object_ids):
    save_keys = ['x','y','z','rotation','horizon']
    all_actionable_positions = []
    for id in object_ids:
        event = this_controller.step(
            action="GetInteractablePoses",
            objectId=id,
            standings=[True],
            horizons=[-30, 30],
        )
        available_positions = event.metadata["actionReturn"]
        for a in available_positions:
            if a in all_actionable_positions: continue
            def overlapping(a):
                for s in all_actionable_positions:
                    sdict = {k:v for k,v in zip(save_keys,s)}
                    if np.linalg.norm([a['x']-sdict['x'], a['z']-sdict['z']]) < 1 and \
                        a['rotation'] == sdict['rotation'] and \
                        a['horizon'] == sdict['horizon']: return True
                return False
            if not overlapping(a): all_actionable_positions.append(tuple([a[k] for k in save_keys]))
    
    all_actionable_positions = [{k:v for k,v in zip(save_keys,a)} for a in all_actionable_positions]
    return all_actionable_positions

def make_mask(bbox, img_size):
    xmin, xmax, ymin, ymax = bbox
    mask = np.ones(img_size, dtype=np.uint8)
    mask[:xmin,:] = 0
    mask[xmax:,:] = 0
    mask[:,:ymin] = 0
    mask[:,ymax:] = 0
    return mask


for n_house in range(10000):
    n_house = 1
    house_id = "{:05d}".format(n_house)
    print(f"Processing house {house_id}...")

    house = dataset_procthor["train"][n_house]
    controller.reset(scene=house,
                    renderInstanceSegmentation=True)

    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]

    object_locations, object_ids = get_obj_loc_and_ids(house)

    all_actionable_positions = get_useful_pose_set(controller, object_ids[0]+object_ids[1])

    def hash(arr):
        arr = arr.astype(dtype=np.uint32)
        return (arr[:,:,0]*1000*1000)+(arr[:,:,1]*1000)+(arr[:,:,2])

    def unhash(arr, return_type='array'):
        r = int(arr // (1000*1000))
        g = int((arr % (1000*1000)) // 1000)
        b = int(arr % (1000))
        if return_type == 'array':
            return np.stack([r,g,b], axis=-1)
        elif return_type == 'tuple':
            return (r,g,b)
        else:
            assert False
        
    os.makedirs(f'data/index', exist_ok=True)
    os.makedirs(f'data/house_images', exist_ok=True)
    os.makedirs(f'data/images/{house_id}')
    os.makedirs(f'data/bboxes/{house_id}')

    xs = [rp["x"] for rp in reachable_positions]
    zs = [rp["z"] for rp in reachable_positions]

    xo = [rp["x"] for rp in object_locations[0]]
    zo = [rp["z"] for rp in object_locations[0]]

    xo2 = [rp["x"] for rp in object_locations[1]]
    zo2 = [rp["z"] for rp in object_locations[1]]

    xa = [rp["x"] for rp in all_actionable_positions]
    za = [rp["z"] for rp in all_actionable_positions]


    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, zs)
    ax.scatter(xo, zo, c='r')
    ax.scatter(xo2, zo2, c='y')
    ax.scatter(xa, za, c='c')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("Reachable Positions in the Scene")
    ax.set_aspect("equal")
    get_top_down_frame().save(f'data/house_images/{house_id}_top_down.jpg')
    fig.savefig(f'data/house_images/{house_id}_poses.jpg')

    obj_to_poseid = {}
    for pose in all_actionable_positions:
        horizon = 0
        standing = True
        pose_id = f"{pose['x']}_{pose['z']}_{pose['rotation']}_{horizon}_{standing}"
        # print("Teleporting the agent to", pose)
        event = controller.step(action="Teleport",
                                position=dict(x=pose['x'], y=pose['y'], z=pose['z']),
                                rotation=dict(x=0, y=pose['rotation'], z=0),
                                horizon=horizon,
                                standing=standing
                                )
        image = Image.fromarray(event.frame)
        color_hashes = hash(event.instance_segmentation_frame)
        colors_unique = np.unique(color_hashes)
        obj_ids_in_frame = ([event.color_to_object_id[unhash(c, 'tuple')] for c in colors_unique if unhash(c, 'tuple') in event.color_to_object_id])
        for obj in obj_ids_in_frame:
            if obj not in obj_to_poseid: obj_to_poseid[obj] = [pose_id]
            else: obj_to_poseid[obj].append(pose_id)
        draw = ImageDraw.Draw(image, "RGBA")
        bboxes = {}
        for c in colors_unique:
            obj_id = event.color_to_object_id[unhash(c, 'tuple')] if unhash(c, 'tuple') in event.color_to_object_id else None
            if obj_id is None:
                print(f"     Could not find {unhash(c, 'tuple')}!")
            if obj_id not in object_ids[0]+object_ids[1]:
                print(f"     Skipping {obj_id}!")
                continue
            xmin = np.argwhere(color_hashes == c)[:,0].min()
            xmax = np.argwhere(color_hashes == c)[:,0].max()
            ymin = np.argwhere(color_hashes == c)[:,1].min()
            ymax = np.argwhere(color_hashes == c)[:,1].max()
            bboxes[obj_id] = (xmin, xmax, ymin, ymax)
            draw.rectangle(((ymin, xmin), (ymax, xmax)), outline=(0, 0, 0, 180), width=2)
        if len(bboxes) > 0:
            image.save(f'data/images/{house_id}/{pose_id}.jpg')
            pickle.dump(bboxes, open(f'data/bboxes/{house_id}/{pose_id}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            bbox_masks = {k:make_mask(v, img_size=event.frame.shape[:2]) for k,v in bboxes.items()}
            pickle.dump({'rgb':event.frame, 'masks':bbox_masks}, open(f'data/model/{house_id}/{pose_id}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    json.dump({'obj_to_poseid':obj_to_poseid, 'ids':object_ids[0]+object_ids[1]}, open(f'data/index/{house_id}.json','w'), indent=4)
