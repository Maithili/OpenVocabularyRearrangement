"""Script to train a pytorch MLP model."""

import os
import csv
import datetime
import json
from pathlib import Path
from absl import app, flags

import random
import typing

import clip
import torch
from torch import nn

import constants
import utils

flags.DEFINE_string("user", None, "Name of the user to train the model for.")
flags.DEFINE_string(
    "embeddings_file_path",
    "object_data/embeddings_saved/CLIP_max_DINO_max.pt",
    "Path to the embeddings file.",
)

FLAGS = flags.FLAGS

CLIP_EMBEDDINGS_FILE_PATH = "object_data/clip_text_embeddings.pt"
CLIP_EMBEDDING_DIM = 512

random.seed(3289758934)
torch.manual_seed(3289758934)


def object_placement_to_tensor(
    object_id, room_name, surface_name, tensor_dict, clip_dict
):
    """Function to convert object id to tensor."""

    if object_id not in tensor_dict:
        raise ValueError(f"Object {object_id} not in tensor dict.")
    if "room_" + room_name not in clip_dict:
        raise ValueError(f"Room {room_name} not in clip dict.")
    if "surface_" + surface_name not in clip_dict:
        raise ValueError(f"Surface {surface_name} not in clip dict.")

    object_tensor = tensor_dict[object_id].view(1, -1)
    room_tensor = clip_dict["room_" + room_name].view(1, -1)
    surface_tensor = clip_dict["surface_" + surface_name].view(1, -1)
    return object_tensor, room_tensor, surface_tensor


def generate_batches_from_data(object_surface_tensors, batch_size=16):
    # random.shuffle(object_surface_tensors)
    object_tensors = [d[0] for d in object_surface_tensors]
    room_names = [d[1] for d in object_surface_tensors]
    surface_names = [d[2] for d in object_surface_tensors]

    labels_tensor = (
        torch.tensor([d[3] for d in object_surface_tensors]).to(torch.float32).view(-1)
    )

    batches = []

    for i in range(0, len(object_surface_tensors), batch_size):
        if i + batch_size > len(object_surface_tensors):
            batches.append(
                [
                    torch.concat(object_tensors[i:], dim=0),
                    torch.concat(room_names[i:], dim=0),
                    torch.concat(surface_names[i:], dim=0),
                    labels_tensor[i:],
                ]
            )
        else:
            batches.append(
                [
                    torch.concat(object_tensors[i : i + batch_size], dim=0),
                    torch.concat(room_names[i : i + batch_size], dim=0),
                    torch.concat(surface_names[i : i + batch_size], dim=0),
                    labels_tensor[i : i + batch_size],
                ]
            )

    return batches


class MatchingModelSimple(nn.Module):
    HIDDEN_DIM_NETWORK = 2048

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
    batches: typing.List[typing.Any],
    input_object_dimension: int,
    num_epochs: int,
    learning_rate: float,
    target_ckpt_folder: str = "",
):
    """Function to train the full MLP model"""

    # Initialize the model
    model = MatchingModelSimple(input_object_dimension + 2*CLIP_EMBEDDING_DIM)
    model.cuda()

    # TODO: make train and validation exclusive.
    train_batches = batches
    validation_batches = batches

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-20
    )

    # Train the model
    model.train()
    break_flag = False
    prev_loss = torch.inf
    for epoch in range(num_epochs):
        if break_flag:
            break
        for batch_num, batch in enumerate(train_batches):
            optimizer.zero_grad()
            object_vec, room_vec, surface_vec, label = batch
            object_vec = object_vec.cuda()
            room_vec = room_vec.cuda()
            surface_vec = surface_vec.cuda()
            label = label.cuda()

            pred = model(torch.concat([object_vec, room_vec, surface_vec], dim=1))
            loss = criterion(pred.view(-1), label)
            loss.backward()
            optimizer.step()

        val_loss = 0
        with torch.no_grad():
            for val_batch in validation_batches:
                object_vec, room_vec, surface_vec, label = val_batch
                object_vec = object_vec.cuda()
                room_vec = room_vec.cuda()
                surface_vec = surface_vec.cuda()
                label = label.cuda()

                pred = model(torch.concat([object_vec, room_vec, surface_vec], dim=1))
                loss = criterion(pred.view(-1), label).detach()
                val_loss += loss.item()
        val_loss /= len(validation_batches)

        if abs(val_loss - prev_loss) < 1e-3:
            print(f"Early stopping at epoch {epoch}.")
            break
        prev_loss = float(val_loss)
        print(f"Epoch: {epoch}, Val Loss: {val_loss}")

    # Save the model
    embb_tag = FLAGS.embeddings_file_path.split("/")[-1].split(".")[0]
    torch.save(
        model.state_dict(),
        os.path.join(
            target_ckpt_folder,
            f"model_peruser_{FLAGS.user}_{embb_tag}_model.pth"
        )
    )


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    user_layout_dict = {}
    with open("user_data/user_layout_mapping.txt", "r") as ftxt:
        text_data = ftxt.readlines()
        for row in text_data:
            name_layout_tuple = row.strip().split(",")
            user_layout_dict[name_layout_tuple[0]] = name_layout_tuple[1]
    user_layout = user_layout_dict[FLAGS.user]

    with open("object_data/data_splits.json", "r") as fjson:
        object_split = json.load(fjson)

    with open("data.json", "r") as jsonfile:
        all_user_data = json.load(jsonfile)

    if FLAGS.user not in all_user_data:
        raise ValueError(f"User {FLAGS.user} does not exist.")
    else:
        user_data = all_user_data[FLAGS.user]

    with open(FLAGS.embeddings_file_path, "rb") as f:
        object_tensor_dict = torch.load(f)
    with open(CLIP_EMBEDDINGS_FILE_PATH, "rb") as f:
        clip_text_embeddings_dict = torch.load(f)

    input_object_dimension = list(object_tensor_dict.values())[0].shape[-1]
    print(f"Input object dimension: {input_object_dimension}")

    object_surface_tuples = utils.load_train_data(
        user_data, object_split, user_layout, negative=True
    )
    assert not all(t[3] == 1 for t in object_surface_tuples)

    tensor_data = []
    for object_id, room_name, surface_name, label in object_surface_tuples:
        object_tensor, room_tensor, surface_tensor = object_placement_to_tensor(
            object_id,
            room_name,
            surface_name,
            object_tensor_dict,
            clip_text_embeddings_dict
        )
        tensor_data.append(
            [object_tensor, room_tensor, surface_tensor, label]
        )

    train_batches = generate_batches_from_data(tensor_data, batch_size=16)

    # Create logs folder and train model.
    Path("./logs/modelPerUser").mkdir(parents=True, exist_ok=True)
    train_model(
        train_batches,
        input_object_dimension=input_object_dimension,
        num_epochs=500,
        learning_rate=1e-4,
        target_ckpt_folder="./logs/modelPerUser",
    )


if __name__ == "__main__":
    app.run(main)
