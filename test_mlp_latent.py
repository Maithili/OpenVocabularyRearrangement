"""Script to test a pytorch MLP model."""

import os
import json
from pathlib import Path
import typing
import random

import torch
from torch import nn
from absl import app, flags

import constants
import utils


flags.DEFINE_string("user", None, "Name of the user to test on.")
flags.DEFINE_string(
    "embeddings_file_path",
    "object_data/embeddings_saved/CLIP_max_DINO_max.pt",
    "Path to the embeddings file.",
)
flags.DEFINE_string("checkpoint", None, "Path to the checkpoint file.")

FLAGS = flags.FLAGS
CLIP_EMBEDDINGS_FILE_PATH = "object_data/clip_text_embeddings.pt"

# TODO: this will change with embeddings_file_path.
OBJECT_EMBEDDING_DIM = 512
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


def test_model(
    model,
    z_latent,
    tensor_batches,
    ground_truth_list,
    object_id_list,
    room_surface_tuples,
):
    model.eval()
    prediction_array = None
    for i, (object_vec, room_vec, surface_vec, _) in enumerate(tensor_batches):
        object_vec = object_vec.cuda()
        room_vec = room_vec.cuda()
        surface_vec = surface_vec.cuda()
        z_tensor = torch.concat([z_latent.view(1, -1)] * object_vec.shape[0], dim=0)
        x_embb = torch.cat(
            [
                z_tensor,
                object_vec,
                room_vec,
                surface_vec,
            ],
            dim=1,
        )
        output_probabilities = model(x_embb)
        if prediction_array is None:
            prediction_array = output_probabilities
        else:
            prediction_array = torch.cat(
                [prediction_array, output_probabilities], dim=0
            )
    prediction_array = prediction_array.view(-1, len(room_surface_tuples))

    # Find the accuracy for hits@1.
    prediction_indices_hits1 = (
        torch.argmax(prediction_array, dim=1).detach().cpu().numpy().astype(int)
    )
    predicted_placements_hits1 = [
        room_surface_tuples[i] for i in prediction_indices_hits1
    ]
    prediction_bool_hits1 = []
    for object_id, pred in zip(object_id_list, predicted_placements_hits1):
        ground_truths = [
            (orr_tuple[1], orr_tuple[2])
            for orr_tuple in ground_truth_list
            if orr_tuple[0] == object_id
        ]
        prediction_bool_hits1.append(pred in ground_truths)
        # print(f"{pred in ground_truths} | Object {object_id} predicted {pred}, ground truth {ground_truths}")
    accuracy_hits1 = sum(prediction_bool_hits1) / len(prediction_bool_hits1)

    # Find the accuracy for hits@3.
    prediction_indices_hits3 = (
        torch.argsort(prediction_array, dim=1, descending=True)
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )
    predicted_placements_hits3 = [
        [room_surface_tuples[i] for i in indices]
        for indices in prediction_indices_hits3
    ]
    prediction_bool_hits3 = []
    for object_id, pred_list in zip(object_id_list, predicted_placements_hits3):
        ground_truths = [
            (orr_tuple[1], orr_tuple[2])
            for orr_tuple in ground_truth_list
            if orr_tuple[0] == object_id
        ]
        prediction_bool_hits3.append(any(pred in ground_truths for pred in pred_list))
    accuracy_hist3 = sum(prediction_bool_hits3) / len(prediction_bool_hits3)
    return accuracy_hits1, accuracy_hist3


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

    if FLAGS.user not in all_user_data:
        raise ValueError(f"User {FLAGS.user} does not exist.")
    user_data = all_user_data[FLAGS.user]

    room_surface_tuples = [
        (room_name, surface_name)
        for room_name, surfaces_in_room in constants.HOME_LAYOUTS[
            user_layout_dict[FLAGS.user]
        ].items()
        for surface_name in surfaces_in_room.values()
    ]

    with open(FLAGS.embeddings_file_path, "rb") as f:
        object_tensor_dict = torch.load(f)
    with open(CLIP_EMBEDDINGS_FILE_PATH, "rb") as f:
        clip_text_embeddings_dict = torch.load(f)

    input_object_dimension = list(object_tensor_dict.values())[0].shape[-1]

    train_ground_truth_list = utils.load_train_data(
        user_data, object_split, user_layout_dict[FLAGS.user], negative=False
    )
    assert all(t[3] == 1 for t in train_ground_truth_list)

    train_object_surface_meshgrid = utils.generate_object_surface_meshgrid(
        object_split["train_memorization"] + object_split["train_others"],
        user_layout_dict[FLAGS.user],
    )
    user_tensor_data = []
    for object_id, room_name, surface_name in train_object_surface_meshgrid:
        object_tensor, room_tensor, surface_tensor = object_placement_to_tensor(
            object_id,
            room_name,
            surface_name,
            object_tensor_dict,
            clip_text_embeddings_dict,
        )
        user_tensor_data.append([object_tensor, room_tensor, surface_tensor, -1])
    train_batches = generate_batches_from_data(user_tensor_data, batch_size=32)

    # Find the latent z.
    model = MatchingModelWithLatent(
        input_dim=1 + input_object_dimension + 2 * CLIP_EMBEDDING_DIM
    ).cuda()
    if FLAGS.checkpoint is None:
        raise ValueError("No checkpoint provided.")
    model.load_state_dict(torch.load(FLAGS.checkpoint))

    # Find the latent z.
    z_grid_search = torch.linspace(-1, 1, 201).cuda()
    max_accuracy = -torch.inf
    z_latent = None
    model.eval()
    for z_grid in z_grid_search:
        accuracy, _ = test_model(
            model,
            z_grid,
            train_batches,
            train_ground_truth_list,
            object_split["train_memorization"] + object_split["train_others"],
            room_surface_tuples,
        )
        print(f"z_grid: {z_grid.detach().cpu().numpy()}, accuracy: {accuracy}")
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            z_latent = z_grid
    print(f"Found z_latent: {z_latent}")

    ground_truth_data_dict = utils.load_test_data(
        user_data, object_split, user_layout_dict[FLAGS.user]
    )
    prediction_input_dict = {}
    for key in [
        "train_memorization",
        "test_generalization_cat",
        "test_generalization_subcat",
    ]:
        prediction_input_dict[key] = utils.generate_object_surface_meshgrid(
            object_split[key], user_layout_dict[FLAGS.user]
        )

    prediction_tensor_dict = {}
    for key, test_data in prediction_input_dict.items():
        for object_id, room_name, surface_name in test_data:
            object_tensor, room_tensor, surface_tensor = object_placement_to_tensor(
                object_id,
                room_name,
                surface_name,
                object_tensor_dict,
                clip_text_embeddings_dict,
            )
            if not key in prediction_tensor_dict:
                prediction_tensor_dict[key] = []
            prediction_tensor_dict[key].append(
                [object_tensor, room_tensor, surface_tensor, -1]
            )

    test_batches_dict = {
        key: generate_batches_from_data(tensor_list, batch_size=32)
        for key, tensor_list in prediction_tensor_dict.items()
    }

    results_dict = {}
    results_dict["z_latent"] = str(z_latent.detach().cpu().float())
    results_dict["checkpoint"] = FLAGS.checkpoint
    for key, test_batch in test_batches_dict.items():
        accuracy_hits1, accuracy_hits3 = test_model(
            model,
            z_latent,
            test_batch,
            ground_truth_data_dict[key],
            object_split[key],
            room_surface_tuples,
        )
        results_dict[key] = {
            "accuracy@hits1": accuracy_hits1,
            "accuracy@hits3": accuracy_hits3,
        }
        print(f"Accuracy for {key}: @1 - {accuracy_hits1}, @3 - {accuracy_hits3}")

    Path("results").mkdir(parents=True, exist_ok=True)
    with open(f"results/results_{FLAGS.user}.json", "w") as fjson:
        json.dump(results_dict, fjson)


if __name__ == "__main__":
    app.run(main)
