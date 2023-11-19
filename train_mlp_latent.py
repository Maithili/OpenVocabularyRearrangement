"""Script to train a pytorch MLP model."""

import os
import json
from pathlib import Path
import typing
import random
import matplotlib.pyplot as plt

import numpy as np
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
    tanh_function = nn.Tanh()

    # TODO: validation dataset?
    list_of_users = list(train_batches_dict.keys())
    zlatent_per_user = {}
    for user in list_of_users:
        zlatent_per_user[user] = torch.tensor([0], dtype=torch.float32).cuda()
        zlatent_per_user[user].requires_grad = True
        nn.init.uniform_(zlatent_per_user[user], -1, 1)
        print(f"Initial z for user {user}: {zlatent_per_user[user]}")

    # Initialize the model
    latent_model = MatchingModelWithLatent(
        input_dim=1 + input_object_dimension + 2 * CLIP_EMBEDDING_DIM
    ).cuda()
    # Define the loss function.
    criterion = nn.BCELoss()
    # Define the optimizer.
    train_parameters_shared = list(latent_model.parameters())
    train_parameters_not_shared = [zlatent_per_user[user] for user in list_of_users]
    optimizer = torch.optim.Adam(
            [
                {"params": train_parameters_shared,
                 "lr": learning_rate/len(list_of_users),
                 "weight_decay": 1e-20/len(list_of_users)},
                {"params": train_parameters_not_shared,
                 "lr": learning_rate,
                 "weight_decay": 1e-20},
            ]
        )

    # Train the model
    torch.autograd.set_detect_anomaly(True)
    num_batches_per_user = [len(train_batches_dict[u]) for u in list_of_users]
    print(f"Number of batches per user: {num_batches_per_user}")
    loss_history = []
    prev_loss_per_user = [torch.inf for _ in list_of_users]
    break_flag = False
    for epoch in range(num_epochs):
        if break_flag:
            break
        for batch_num in range(max(num_batches_per_user)):
            optimizer.zero_grad()
            loss_batch_per_user = {}
            for uid, user in enumerate(list_of_users):
                if batch_num >= num_batches_per_user[uid]:
                    loss_batch_per_user[user] = None
                    continue

                object_vec, room_vec, surface_vec, label = train_batches_dict[user][
                    batch_num
                ]
                object_vec = object_vec.cuda()
                room_vec = room_vec.cuda()
                surface_vec = surface_vec.cuda()
                label = label.cuda()

                # Normalize zlatent to [-1, 1] using tanh.
                zlatent_normalized = tanh_function(zlatent_per_user[user])
                zlatent_user_repeat = zlatent_normalized.view(1, -1).repeat(
                    object_vec.shape[0], 1
                )
                if batch_num % 10 == 0:
                    print(f"Z for user {user}: {zlatent_normalized}")
                input_vec = torch.concat(
                    [zlatent_user_repeat, object_vec, room_vec, surface_vec], dim=1
                )
                pred = latent_model(input_vec).view(-1)
                loss_batch_per_user[user] = criterion(pred, label)
                loss_batch_per_user[user].backward()

            optimizer.step()
            current_loss_per_user = [
                loss_batch_per_user[u].item()
                for u in list_of_users
                if loss_batch_per_user[u] is not None
            ]
            if all (
                abs(current_loss_per_user[i] - prev_loss_per_user[i]) < 1e-2
                for i in range(len(current_loss_per_user))
            ) and len(current_loss_per_user) == len(list_of_users):
                print(f"Prev loss: {prev_loss_per_user}")
                print(f"Current loss: {current_loss_per_user}")
                print(f"Early stopping at epoch {epoch}, iter {batch_num}.")
                break_flag = True
                break
            for i, user in enumerate(list_of_users):
                if loss_batch_per_user[user] is not None:
                    prev_loss_per_user[i] = loss_batch_per_user[user].item()
            if batch_num % 10 == 0:
                print(
                    f"Epoch: {epoch}, Iteration: {batch_num}, user {user}, Loss: {prev_loss_per_user}"
                    )
            loss_history.append(list(prev_loss_per_user))

    # Save the model
    embb_tag = FLAGS.embeddings_file_path.split("/")[-1].split(".")[0]
    torch.save(
        latent_model.state_dict(),
        os.path.join(target_ckpt_folder, f"model_exclude_{FLAGS.excluded_user}_embb_{embb_tag}.ckpt"),
    )

    for i, user in enumerate(list_of_users):
        loss_history_user = [l[i] for l in loss_history]
        plt.plot(range(len(loss_history_user)), loss_history_user, label=user)
    plt.legend()
    plt.savefig(
        os.path.join(
            target_ckpt_folder,
            f"loss_exclude_{FLAGS.excluded_user}_embb_{embb_tag}.png",
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
        user: generate_batches_from_data(tensor_list, batch_size=16)
        for user, tensor_list in user_tensor_data.items()
    }

    # Create logs folder and train model.
    Path("./logs/sansLatentNetwork").mkdir(parents=True, exist_ok=True)
    train_model(
        user_train_batches,
        input_object_dimension=input_object_dimension,
        num_epochs=30,
        learning_rate=5e-4,
        target_ckpt_folder="./logs/sansLatentNetwork",
    )


if __name__ == "__main__":
    app.run(main)
