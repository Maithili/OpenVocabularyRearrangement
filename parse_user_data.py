"""Script to read CSV files and return a dictionary of the data."""

import os
import csv
import json
import random
import string

from absl import app, flags

import constants

flags.DEFINE_string(
    "csv_folder", "./user_data", "Path to folder containing CSV files."
)
FLAGS = flags.FLAGS


# TODO: only include object ids that all users have annotated.
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    csv_folder = FLAGS.csv_folder
    if csv_folder is None or not os.path.isdir(csv_folder):
        raise ValueError("Folder path must be specified.")

    csv_files = os.listdir(csv_folder)

    data_dict = {}
    # Load data from each file as a dictionary in csv_files and append to the
    # data variable.
    user_layout_dict = {}
    with open(os.path.join(csv_folder, "user_layout_mapping.txt")) as ftxt:
        text_data = ftxt.readlines()
        for row in text_data:
            name_layout_tuple = row.strip().split(",")
            user_layout_dict[name_layout_tuple[0]]=name_layout_tuple[1]
    for csv_file in csv_files:
        # # Generate random alphanumeric string.
        # anonymized_id = "".join(
        #     random.choices(
        #         string.ascii_uppercase + string.digits, k=10
        #     )
        # )
        if csv_file.endswith(".txt"):
            continue
        elif csv_file.endswith(".csv"):
            data = []
            user_name = csv_file.split(".")[0].split("_")[-1]
            if user_name not in user_layout_dict:
                raise ValueError(
                    f"Invalid user {user_name}."
                )
            row_list = csv.reader(open(os.path.join(csv_folder, csv_file)))
        else:
            raise ValueError(
                f"File {csv_file} is neither a .csv or .txt file."
            )

        # TODO: Anonymize.
        anonymized_id = user_name
        print(f"User {user_name} with anonymized id {anonymized_id}")
        for row_num, row in enumerate(row_list):
            if row_num < 4:
                continue
            if not row[2] or not row[3]:
                # TODO: how to handle missing annotations?
                print(
                    f"Row {row_num} in {csv_file} does not have any placements"
                )
                continue
            user_item_dict = {
                "object_id": row[0],
            }
            layout_user = user_layout_dict[user_name]
            rooms = [row[2], row[4], row[6]]
            surfaces = [row[3], row[5], row[7]]
            surfaces = [s.strip(")").strip("(") for s in surfaces]
            i = 1
            skip = False
            for room, surface in zip(rooms, surfaces):
                if not room or not surface:
                    continue
                if room not in constants.HOME_LAYOUTS[layout_user]:
                    raise ValueError(
                        f"Room {room} in file {csv_file} does not match constants."
                    )
                user_item_dict[f"room_{i}"] = room
                if surface not in constants.HOME_LAYOUTS[layout_user][room]:
                    print(
                        f"Surface {surface} in file {csv_file} does not match constants."
                    )
                    # Skip surface names that don't match constants.
                    skip = True
                    # # Custom surface, just use whatever the user says.
                    # user_item_dict[f"surface_{i}"] = surface.lower()
                else:
                    user_item_dict[f"surface_{i}"] = (
                        constants.HOME_LAYOUTS[layout_user][room][surface]
                    )
                i += 1
            if not skip:
                data.append(user_item_dict)

        data_dict[anonymized_id] = data
    with open("data.json", "w") as fjson:
        json.dump(data_dict, fjson, indent=4)

if __name__ == "__main__":
    app.run(main)
