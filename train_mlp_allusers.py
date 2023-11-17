"""Script to train a pytorch MLP model."""

import os
import csv
import datetime
import json
from pathlib import Path
import random
import typing

import clip
import torch
from torch import nn

import constants

INPUT_DIM = 512*3
random.seed(3289758934)

DATETIMESTR = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")

def _text_to_clip():
    """Function to convert text to clip embeddings."""

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    def _generate_clip_text_embeddings(text_inputs):
        # Encode the text inputs
        text_embeddings = []
        for text in text_inputs:
            text_tensor = clip.tokenize(text).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(text_tensor)
                text_embeddings.append(text_embedding)

        if len(text_embedding.shape) == 1:
            text_embeddings = torch.stack(text_embeddings)
        else:
            assert text_embedding.shape[0] == 1
            text_embeddings = torch.cat(text_embeddings, dim=0)

        return text_embeddings

    return _generate_clip_text_embeddings


def _json_object_to_tensor(object_surface_data, batch_size=16):
    object_names = [d[0] for d in object_surface_data]
    room_names = [d[1] for d in object_surface_data]
    surface_names = [d[2] for d in object_surface_data]

    clip_generator = _text_to_clip()
    object_tensors = clip_generator(object_names).to(torch.float32)
    room_tensors = clip_generator(room_names).to(torch.float32)
    surface_tensors = clip_generator(surface_names).to(torch.float32)

    labels_tensor = torch.tensor(
        [d[3] for d in object_surface_data]
    ).to(torch.float32).view(-1)
    batches = []

    for i in range(0, len(object_surface_data), batch_size):
        if i + batch_size > len(object_surface_data):
            batches.append(
                [
                    object_tensors[i:],
                    room_tensors[i:],
                    surface_tensors[i:],
                    labels_tensor[i:],
                ]
            )
        else:
            batches.append(
                [
                    object_tensors[i : i + batch_size],
                    room_tensors[i : i + batch_size],
                    surface_tensors[i : i + batch_size],
                    labels_tensor[i : i + batch_size],
                ]
            )

    return batches


class MatchingModelSimple(nn.Module):
    HIDDEN_DIM_NETWORK = 2048
    OUTPUT_DIM_LATENT = 1

    def __init__(self, input_dim=512):
        super().__init__()

        # Layers to learn the object-surface matching probability.
        self.fc1 = nn.Linear(input_dim, self.HIDDEN_DIM_NETWORK)
        self.fc2 = nn.Linear(self.HIDDEN_DIM_NETWORK, self.HIDDEN_DIM_NETWORK)
        self.fc3 = nn.Linear(self.HIDDEN_DIM_NETWORK, 1)
        # Activation functions.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_embb):
        x_embb = self.relu(self.fc1(x_embb))
        x_embb = self.relu(self.fc2(x_embb))
        probability_tensor = self.sigmoid(self.fc3(x_embb))
        return probability_tensor


def train_model(
    train_data: typing.List[typing.Any],
    num_epochs: int,
    learning_rate: float,
    model_ckpt: typing.Optional[str] = None,
    target_ckpt_folder: str = "",
):
    """Function to train the full MLP model"""

    # Initialize the model
    model = MatchingModelSimple(INPUT_DIM)
    if model_ckpt is not None:
        model.load_state_dict(torch.load(model_ckpt))
    model = model.cuda().float()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        model.train()
        for i, (object_vec, room_vec, surface_vec, label) in enumerate(train_data):
            object_vec = object_vec.cuda()
            room_vec = room_vec.cuda()
            surface_vec = surface_vec.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            pred = model(torch.concat([object_vec, room_vec, surface_vec], dim=1))
            loss = criterion(pred.view(-1), label)
            loss.backward()
            optimizer.step()

            if i % 15 == 0:
                print(
                    "Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, loss.item())
                )

    # Save the model.
    torch.save(
        model.state_dict(),
        os.path.join(target_ckpt_folder, f"{DATETIMESTR}_ALL_model.pth")
    )


def main():

    user_layout_dict = {}
    with open("user_data/user_layout_mapping.txt", "r") as ftxt:
        text_data = ftxt.readlines()
        for row in text_data:
            name_layout_tuple = row.strip().split(",")
            user_layout_dict[name_layout_tuple[0]] = name_layout_tuple[1]

    with open("object_data/data_splits.json", "r") as fjson:
        object_split = json.load(fjson)

    with open("data.json", "r") as jsonfile:
        all_user_data = json.load(jsonfile)

    # Load object names.
    with open("object_data/object_categories_names.csv", "r") as fcsv:
        object_names_list = csv.DictReader(fcsv)
        object_names_dict = {}
        for obj_dict in object_names_list:
            object_names_dict[obj_dict["ObjectId"]] = obj_dict["CaptionsHandWritten"]

    # Load train data.
    train_object_ids = object_split["train_memorization"] + object_split["train_others"]
    train_object_surface_data = []
    for user in all_user_data:
        layout = user_layout_dict[user]
        user_data = all_user_data[user]
        for obj_dict in user_data:
            if obj_dict["object_id"] not in train_object_ids:
                continue

            object_name = object_names_dict[obj_dict["object_id"]]

            # Add positive samples.
            positive_rooms_list = []
            positive_surfaces_list = []
            for i in range(1, 4):
                if f"room_{i}" in obj_dict and f"surface_{i}" in obj_dict:
                    train_object_surface_data.append(
                        [object_name, obj_dict[f"room_{i}"], obj_dict[f"surface_{i}"], 1]
                    )
                    if obj_dict[f"room_{i}"] not in positive_rooms_list:
                        positive_rooms_list.append(obj_dict[f"room_{i}"])
                    if obj_dict[f"surface_{i}"] not in positive_surfaces_list:
                        positive_surfaces_list.append(obj_dict[f"surface_{i}"])

            # Select random room and surface for negative samples.
            # Random room and random surface.
            # TODO: change negative sampling to ensure that negative samples
            # do not conflict with another user's preference.
            excluded_rooms = [
                r for r in list(constants.HOME_LAYOUTS[layout].keys()) if r not in positive_rooms_list
            ]
            random_room = random.sample(excluded_rooms, 1)[0]
            random_surface_outroom = random.sample(
                list(constants.HOME_LAYOUTS[layout][random_room].values()), 1
            )
            train_object_surface_data.append(
                [object_name, random_room, random_surface_outroom[0], 0]
            )
            # Correct room, random surface.
            inroom = random.sample(positive_rooms_list, 1)[0]
            inroom_negative_surfaces = [
                s
                for s in list(constants.HOME_LAYOUTS[layout][inroom].values())
                if s not in positive_surfaces_list
            ]
            if not inroom_negative_surfaces:
                continue
            random_surface_inroom = random.sample(inroom_negative_surfaces, 1)
            train_object_surface_data.append(
                [object_name, inroom, random_surface_inroom[0], 0]
            )

    train_batches = _json_object_to_tensor(train_object_surface_data)

    # Create logs folder and train model.
    Path("./logs").mkdir(parents=True, exist_ok=True)
    train_model(
        train_batches,
        num_epochs=50,
        learning_rate=1e-4,
        model_ckpt=None,
        target_ckpt_folder="./logs",
    )


if __name__ == "__main__":
    main()
