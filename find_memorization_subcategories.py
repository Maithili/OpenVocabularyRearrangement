import csv
import json

fcsv = open("object_data/object_categories_names.csv", "r")
reader = csv.DictReader(fcsv)

object_data = [row for row in reader]

category_to_object_id = {}
subcategory_to_object_id = {}
# Find all unique categories and subcategories.
for row in object_data:
    if row["Category"] not in category_to_object_id:
        category_to_object_id[row["Category"]] = [row["ObjectId"]]
    else:
        category_to_object_id[row["Category"]].append(row["ObjectId"])
    if row["Subcategory"] not in subcategory_to_object_id:
        subcategory_to_object_id[row["Subcategory"]] = [row["ObjectId"]]
    else:
        subcategory_to_object_id[row["Subcategory"]].append(row["ObjectId"])

fdata = open("data.json", "r")
all_user_data = json.load(fdata)

for user_name, user_data in all_user_data.items():
    print(f"User {user_name}")

    user_data_dict = {}
    for data in user_data:
        object_id = data["object_id"]
        del data["object_id"]
        user_data_dict[object_id] = data

    for subcat, object_id_list in subcategory_to_object_id.items():
        # Check whether all objects in object_id_list are identical in user_data_dict.

        for first_index, first_object_id in enumerate(object_id_list):
            # Skip missing object ids.
            if first_object_id not in user_data_dict:
                continue
            first_object_id_placement = user_data_dict[first_object_id]
            first_placement_rooms = [
                first_object_id_placement[f"room_{i}"]
                for i in range(1, 4)
                if f"room_{i}" in first_object_id_placement
            ]
            first_placement_surfaces = [
                first_object_id_placement[f"surface_{i}"]
                for i in range(1, 4)
                if f"surface_{i}" in first_object_id_placement
            ]

            for second_object_id in object_id_list[first_index + 1 :]:
                # Skip missing object ids.
                if second_object_id not in user_data_dict:
                    continue
                second_object_id_placement = user_data_dict[second_object_id]
                second_placement_rooms = [
                    second_object_id_placement[f"room_{i}"]
                    for i in range(1, 4)
                    if f"room_{i}" in second_object_id_placement
                ]
                second_placement_surfaces = [
                    second_object_id_placement[f"surface_{i}"]
                    for i in range(1, 4)
                    if f"surface_{i}" in second_object_id_placement
                ]
                if not any(
                    [room in second_placement_rooms for room in first_placement_rooms]
                ):
                    print(
                        f"Object {first_object_id} in subcategory {subcat} does not have the same room placement as the {second_object_id}."
                    )
                    print("First: ", first_placement_rooms)
                    print("Second: ", second_placement_rooms, "\n")
                if not any(
                    [
                        surface in second_placement_surfaces
                        for surface in first_placement_surfaces
                    ]
                ):
                    print(
                        f"Object {first_object_id} in subcategory {subcat} does not have the same surface placement as the {second_object_id}."
                    )
                    print("First: ", first_placement_surfaces)
                    print("Second: ", second_placement_surfaces, "\n")
