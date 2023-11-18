"""Utility functions for the project."""

import random
import constants

random.seed(3289758934)

def load_test_data(user_data, object_split, layout_key):
    # Load train data.
    object_surface_data_dict = {}
    for data_label, object_ids in zip([
        "train_memorization",
        "test_generalization_cat",
        "test_generalization_subcat"
    ], [
        object_split["train_memorization"],
        object_split["test_generalization_cat"],
        object_split["test_generalization_subcat"]
    ]):
        object_surface_data = []
        for obj_dict in user_data:
            # TODO: better way to skip objects not in the training set?
            if obj_dict["object_id"] not in object_ids:
                continue

            object_id = obj_dict["object_id"]

            # Add positive samples.
            positive_rooms_list = []
            positive_surfaces_list = []
            for i in range(1, 4):
                if f"room_{i}" in obj_dict and f"surface_{i}" in obj_dict:
                    object_surface_data.append(
                        [object_id, obj_dict[f"room_{i}"], obj_dict[f"surface_{i}"], 1]
                    )
                    if obj_dict[f"room_{i}"] not in positive_rooms_list:
                        positive_rooms_list.append(obj_dict[f"room_{i}"])
                    if obj_dict[f"surface_{i}"] not in positive_surfaces_list:
                        positive_surfaces_list.append(obj_dict[f"surface_{i}"])

            # Select random room and surface for negative samples.
            # Random room and random surface.
            excluded_rooms = [
                r for r in list(constants.HOME_LAYOUTS[layout_key].keys())
                if r not in positive_rooms_list
            ]
            random_room = random.sample(excluded_rooms, 1)[0]
            random_surface_outroom = random.sample(
                list(constants.HOME_LAYOUTS[layout_key][random_room].values()), 1
            )
            object_surface_data.append(
                [object_id, random_room, random_surface_outroom[0], 0]
            )
            # Correct room, random surface.
            inroom = random.sample(positive_rooms_list, 1)[0]
            inroom_negative_surfaces = [
                s
                for s in list(constants.HOME_LAYOUTS[layout_key][inroom].values())
                if s not in positive_surfaces_list
            ]
            if not inroom_negative_surfaces:
                continue
            random_surface_inroom = random.sample(inroom_negative_surfaces, 1)
            object_surface_data.append(
                [object_id, inroom, random_surface_inroom[0], 0]
            )
        object_surface_data_dict[data_label] = object_surface_data

    return object_surface_data_dict


def load_train_data(user_data, object_split, layout_key, negative=True):
    # Load train data.
    train_object_ids = (
        object_split["train_memorization"] + object_split["train_others"]
    )
    positive_object_placements = []
    negative_object_placements = []
    for obj_dict in user_data:
        # TODO: better way to skip objects not in the training set?
        if obj_dict["object_id"] not in train_object_ids:
            continue

        object_id = obj_dict["object_id"]

        # Add positive samples.
        positive_rooms_list = []
        positive_surfaces_list = []
        for i in range(1, 4):
            if f"room_{i}" in obj_dict and f"surface_{i}" in obj_dict:
                positive_object_placements.append(
                    [object_id, obj_dict[f"room_{i}"], obj_dict[f"surface_{i}"], 1]
                )
                if obj_dict[f"room_{i}"] not in positive_rooms_list:
                    positive_rooms_list.append(obj_dict[f"room_{i}"])
                if obj_dict[f"surface_{i}"] not in positive_surfaces_list:
                    positive_surfaces_list.append(obj_dict[f"surface_{i}"])

        if negative:

            # Select random room and surface for negative samples.
            # Random room and random surface.
            excluded_rooms = [
                r for r in list(constants.HOME_LAYOUTS[layout_key].keys())
                if r not in positive_rooms_list
            ]
            random_room = random.sample(excluded_rooms, 1)[0]
            random_surface_outroom = random.sample(
                list(constants.HOME_LAYOUTS[layout_key][random_room].values()), 1
            )
            negative_object_placements.append(
                [object_id, random_room, random_surface_outroom[0], 0]
            )
            # Correct room, random surface.
            inroom = random.sample(positive_rooms_list, 1)[0]
            inroom_negative_surfaces = [
                s
                for s in list(constants.HOME_LAYOUTS[layout_key][inroom].values())
                if s not in positive_surfaces_list
            ]
            if not inroom_negative_surfaces:
                continue
            random_surface_inroom = random.sample(inroom_negative_surfaces, 1)
            negative_object_placements.append(
                [object_id, inroom, random_surface_inroom[0], 0]
            )
    # Add positive and negative samples to a single list and shuffle.
    combined_object_placements = positive_object_placements + negative_object_placements
    random.shuffle(combined_object_placements)
    return combined_object_placements


def generate_object_surface_meshgrid(object_id_list, user_layout):
    """Returns a meshgrid over object ids - room names - surface names."""
    object_surface_meshgrid = []
    for object_id in object_id_list:
        for room in constants.HOME_LAYOUTS[user_layout].keys():
            for surface in constants.HOME_LAYOUTS[user_layout][room].values():
                # print(f"Object {object_id}, room {room}, surface {surface}")
                object_surface_meshgrid.append([object_id, room, surface])
    return object_surface_meshgrid
