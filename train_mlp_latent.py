"""Script to train a pytorch MLP model."""

import os
import json
from pathlib import Path
import typing
import random

import torch
from torch import nn
from absl import app, flags

import utils

flags.DEFINE_string("excluded_user", None, "Name of the user to not train on.")
flags.DEFINE_string(
    "embeddings_file_path",
    "object_data/embeddings_saved/CLIP_max_DINO_max.pt",
    "Path to the embeddings file.",
)
FLAGS = flags.FLAGS

CLIP_EMBEDDINGS_FILE_PATH = "object_data/clip_text_embeddings.pt"
CLIP_EMBEDDING_DIM = 512

random.seed(3289758934)


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


class ZNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.tanh(self.fc3(z))
        return z


class MatchingModelWithLatent(nn.Module):
    HIDDEN_DIM_NETWORK = 2048

    def __init__(self, input_dim: int):
        super().__init__()

        # Layers to learn the object-surface matching probability.
        self.fc1 = nn.Linear(input_dim, self.HIDDEN_DIM_NETWORK)
        self.fc2 = nn.Linear(self.HIDDEN_DIM_NETWORK, self.HIDDEN_DIM_NETWORK)
        self.fc3 = nn.Linear(self.HIDDEN_DIM_NETWORK, 1)
        # Activation functions.
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_embb):
        x_network = x_embb
        x_network = self.relu(self.fc1(x_network))
        x_network = self.relu(self.fc2(x_network))
        x_network = self.sigmoid(self.fc3(x_network))
        return x_network


def train_model(
    train_batches_dict: typing.Dict[str, typing.Any],
    input_object_dimension: int,
    num_epochs: int,
    learning_rate: float,
    target_ckpt_folder: str = "",
):
    # TODO: validation dataset?
    users = list(train_batches_dict.keys())
    znetwork_per_user = {user: ZNetwork().cuda() for user in users}

    # Initialize the model
    latent_model = MatchingModelWithLatent(
        input_dim=1 + input_object_dimension + 2 * CLIP_EMBEDDING_DIM
    ).cuda()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer_per_user = {
        user: torch.optim.Adam(
            list(znetwork_per_user[user].parameters())
            + list(latent_model.parameters()),
            lr=learning_rate,
        )
        for user in users
    }

    # Train the model
    torch.autograd.set_detect_anomaly(True)
    num_batches_per_user = [len(train_batches_dict[u]) for u in users]
    print(f"Number of batches per user: {num_batches_per_user}")
    for epoch in range(num_epochs):
        for batch_num in range(max(num_batches_per_user)):
            # TODO: how to train multiple networks using single buffer?
            loss_batch_per_user = {}
            for uid, user in enumerate(users):
                if batch_num >= num_batches_per_user[uid]:
                    continue
                # Generate model outputs for each user.
                optimizer_per_user[user].zero_grad()

                object_vec, room_vec, surface_vec, label = train_batches_dict[user][
                    batch_num
                ]
                object_vec = object_vec.cuda()
                room_vec = room_vec.cuda()
                surface_vec = surface_vec.cuda()
                label = label.cuda()

                # Generate the latent vector for each user from uniform
                # distribution samples.
                z_input = torch.rand(size=(len(label), 1)).cuda()
                z_output = znetwork_per_user[user](z_input)
                if batch_num % 10 == 0:
                    print(f"Mean z for user {user}: {torch.mean(z_output).cpu()}")

                input_vec = torch.concat(
                    [z_output, object_vec, room_vec, surface_vec], dim=1
                )
                pred = latent_model(input_vec).view(-1)
                loss_batch_per_user[user] = criterion(pred, label)
                loss_batch_per_user[user].backward()

                optimizer_per_user[user].step()

                if batch_num % 10 == 0:
                    print(
                        "Epoch: {}, Iteration: {}, Loss: {}".format(
                            epoch, batch_num, [loss_batch_per_user[user]]
                        )
                    )

    # Save the model
    torch.save(
        latent_model.state_dict(),
        os.path.join(target_ckpt_folder, f"model_exclude_{FLAGS.excluded_user}.ckpt"),
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

    with open("object_data/data_splits.json", "r") as fjson:
        object_split = json.load(fjson)

    with open("data.json", "r") as jsonfile:
        all_user_data = json.load(jsonfile)

    if FLAGS.excluded_user not in all_user_data:
        raise ValueError(f"User {FLAGS.excluded_user} does not exist.")
    else:
        print(f"Skipping user {FLAGS.excluded_user}.")
        del all_user_data[FLAGS.excluded_user]

    user_object_surface_tuples = {
        user: utils.load_train_data(
            all_user_data[user], object_split, user_layout_dict[user]
        )
        for user in all_user_data
    }

    with open(FLAGS.embeddings_file_path, "rb") as f:
        object_tensor_dict = torch.load(f)
    with open(CLIP_EMBEDDINGS_FILE_PATH, "rb") as f:
        clip_text_embeddings_dict = torch.load(f)

    input_object_dimension = list(object_tensor_dict.values())[0].shape[-1]
    print(f"Input object dimension: {input_object_dimension}")

    user_tensor_data = {}
    for user in user_object_surface_tuples:
        user_tensor_data[user] = []
        for object_id, room_name, surface_name, label in user_object_surface_tuples[
            user
        ]:
            object_tensor, room_tensor, surface_tensor = object_placement_to_tensor(
                object_id,
                room_name,
                surface_name,
                object_tensor_dict,
                clip_text_embeddings_dict,
            )
            user_tensor_data[user].append(
                [object_tensor, room_tensor, surface_tensor, label]
            )

    user_train_batches = {
        user: generate_batches_from_data(tensor_list, batch_size=8)
        for user, tensor_list in user_tensor_data.items()
    }

    # Create logs folder and train model.
    Path("./logs").mkdir(parents=True, exist_ok=True)
    train_model(
        user_train_batches,
        input_object_dimension=input_object_dimension,
        num_epochs=25,
        learning_rate=1e-4,
        target_ckpt_folder="./logs",
    )


if __name__ == "__main__":
    app.run(main)
