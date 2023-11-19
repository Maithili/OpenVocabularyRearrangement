"""Script to train a pytorch MLP model."""

import os
import csv
import datetime
import json
from pathlib import Path
import random
import typing
from absl import app, flags
import pickle as pkl
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
flags.DEFINE_string("checkpoint", None, "Path to the checkpoint file.")

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


def test_model(
    model,
    tensor_batches,
    ground_truth_list,
    object_id_list,
    room_surface_tuples
):
    """Function to test the full MLP model"""

    model.eval()
    predictions = None
    for i, (object_vec, room_vec, surface_vec, _) in enumerate(tensor_batches):
        object_vec = object_vec.cuda()
        room_vec = room_vec.cuda()
        surface_vec = surface_vec.cuda()
        preds = model(
            torch.concat([object_vec, room_vec, surface_vec], dim=1)
        ).view(-1)

        if predictions is None:
            predictions = preds
        else:
            predictions = torch.concat([predictions, preds], dim=0)

    prediction_array = predictions.view(-1, len(room_surface_tuples))

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
        [room_surface_tuples[i] for i in indices[:3]]
        for indices in prediction_indices_hits3
    ]
    prediction_bool_hits3 = []
    for object_id, pred_list in zip(object_id_list, predicted_placements_hits3):
        ground_truths = [
            (orr_tuple[1], orr_tuple[2])
            for orr_tuple in ground_truth_list
            if orr_tuple[0] == object_id
        ]
        # print("Predictions:")
        # for p in pred_list:
        #     print(p)
        # print("Ground truths:")
        # for gt in ground_truths:
        #     print(gt)
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
    user_layout = user_layout_dict[FLAGS.user]

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
    model = MatchingModelSimple(
        input_dim= input_object_dimension + 2 * CLIP_EMBEDDING_DIM
    ).cuda()
    if FLAGS.checkpoint is None:
        raise ValueError("No checkpoint provided.")
    model.load_state_dict(torch.load(FLAGS.checkpoint))

    ground_truth_data_dict = utils.load_test_data(
        user_data, object_split, user_layout
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
    for key, test_batch in test_batches_dict.items():
        accuracy_hits1, accuracy_hits3 = test_model(
            model,
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

    embb_tag = FLAGS.embeddings_file_path.split("/")[-1].split(".")[0]
    Path(f"results/modelPerUser").mkdir(parents=True, exist_ok=True)
    with open(
        f"results/modelPerUser/results_{FLAGS.user}_{embb_tag}.json", "w"
        ) as fjson:
        json.dump(results_dict, fjson, indent=4)


if __name__ == "__main__":
    app.run(main)
