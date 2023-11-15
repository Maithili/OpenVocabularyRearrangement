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

flags.DEFINE_string("user", None, "Name of the user to train the model for.")
flags.DEFINE_string("checkpoint", None, "Path to checkpoint.")

FLAGS = flags.FLAGS
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


def test_model(
    test_data: typing.List[typing.Any],
    model_ckpt: typing.Optional[str],
):
    """Function to train the full MLP model"""

    # Initialize the model
    model = MatchingModelSimple(INPUT_DIM)
    model.load_state_dict(torch.load(model_ckpt))
    model = model.cuda().float()

    model.eval()
    predictions = []
    for i, (object_vec, room_vec, surface_vec, _) in enumerate(test_data):
        object_vec = object_vec.cuda()
        room_vec = room_vec.cuda()
        surface_vec = surface_vec.cuda()
        pred = model(
            torch.concat([object_vec, room_vec, surface_vec], dim=1)
        ).view(-1)
        for p in pred:
            print(p.detach().cpu().numpy())
            predictions.append(p.detach().cpu().numpy())

    return predictions

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    with open("object_data/data_splits.json", "r") as fjson:
        object_split = json.load(fjson)

    with open("data.json", "r") as jsonfile:
        all_user_data = json.load(jsonfile)

    if FLAGS.user not in all_user_data:
        raise ValueError(f"User {FLAGS.user} does not exist.")
    user_data = all_user_data[FLAGS.user]

    # Load object names.
    with open("object_data/object_categories_names.csv", "r") as fcsv:
        object_names_list = csv.DictReader(fcsv)
        object_names_dict = {}
        for obj_dict in object_names_list:
            object_names_dict[obj_dict["ObjectId"]] = obj_dict["CaptionsHandWritten"]

    # Load train data.
    test_memorization_object_ids = object_split["train_memorization"]
    test_generalization_cat_object_ids = object_split["test_generalization_cat"]
    test_generalization_subcat_object_ids = object_split["test_generalization_subcat"]

    test_memorization_data = []
    test_generalization_cat_data = []
    test_generalization_subcat_data = []


    for obj_dict in user_data:
        if obj_dict["object_id"] not in test_memorization_object_ids + test_generalization_cat_object_ids + test_generalization_subcat_object_ids:
            continue
        object_name = object_names_dict[obj_dict["object_id"]]

        placements = []
        for i in range(1, 4):
            if f"room_{i}" in obj_dict and f"surface_{i}" in obj_dict:
                placements.append(
                    [object_name, obj_dict[f"room_{i}"], obj_dict[f"surface_{i}"], 1]
                )

        if obj_dict["object_id"] in test_memorization_object_ids:
            test_memorization_data.extend(placements)
        elif obj_dict["object_id"] in test_generalization_cat_object_ids:
            test_generalization_cat_data.extend(placements)
        elif obj_dict["object_id"] in test_generalization_subcat_object_ids:
            test_generalization_subcat_data.extend(placements)

    # Create logs folder and train model.
    Path("./results").mkdir(parents=True, exist_ok=True)

    checkpoint = FLAGS.checkpoint

    prediction_results = {
        "memorization": [],
        "generalization_cat": [],
        "generalization_subcat": [],
    }

    # Test memorization.
    print("Testing memorization")
    test_memorization_batches = _json_object_to_tensor(test_memorization_data)
    memorization_predictions = test_model(
        test_memorization_batches,
        model_ckpt=checkpoint,
    )
    for example, pred in zip(test_memorization_data, memorization_predictions):
        prediction_results["memorization"].append(
            {
                "object_name": example[0],
                "room_name": example[1],
                "surface_name": example[2],
                "prediction": pred,
            }
        )

    # Test generalization by category.
    print("Testing generalization by category")
    test_generalization_cat_batches = _json_object_to_tensor(test_generalization_cat_data)
    generalization_cat_predictions = test_model(
        test_generalization_cat_batches,
        model_ckpt=checkpoint,
    )
    for example, pred in zip(test_generalization_cat_data, generalization_cat_predictions):
        prediction_results["generalization_cat"].append(
            {
                "object_name": example[0],
                "room_name": example[1],
                "surface_name": example[2],
                "prediction": pred,
            }
        )

    # Test generalization by subcategory.
    print("Testing generalization by subcategory")
    test_generalization_subcat_batches = _json_object_to_tensor(test_generalization_subcat_data)
    generalization_subcat_predictions = test_model(
        test_generalization_subcat_batches,
        model_ckpt=checkpoint,
    )
    for example, pred in zip(test_generalization_subcat_data, generalization_subcat_predictions):
        prediction_results["generalization_subcat"].append(
            {
                "object_name": example[0],
                "room_name": example[1],
                "surface_name": example[2],
                "prediction": pred,
            }
        )

    Path("./results").mkdir(parents=True, exist_ok=True)
    # Save results.
    with open(f"results/{DATETIMESTR}_{FLAGS.user}.pkl", "wb") as fpkl:
        pkl.dump(prediction_results, fpkl)


if __name__ == "__main__":
    app.run(main)
