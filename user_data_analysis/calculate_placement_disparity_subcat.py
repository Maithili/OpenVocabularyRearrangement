import csv
import json
import numpy as np
import matplotlib.pyplot as plt

user_layout_dict = {}
with open("../user_data/user_layout_mapping.txt", "r") as ftxt:
    text_data = ftxt.readlines()
    for row in text_data:
        name_layout_tuple = row.strip().split(",")
        user_layout_dict[name_layout_tuple[0]] = name_layout_tuple[1]

# Load user data from data.json.
with open("../data.json", "r") as fjson:
    user_data_dict = json.load(fjson)

object_ids = [
    x["object_id"]
    for x in user_data_dict[list(user_data_dict.keys())[0]]
]

fcsv = open("../object_data/object_categories_names.csv", "r")
reader = csv.DictReader(fcsv)

subcategory_to_object_id = {}
for row in reader:
    if row["Subcategory"] not in subcategory_to_object_id:
        subcategory_to_object_id[row["Subcategory"]] = [row["ObjectId"]]
    else:
        subcategory_to_object_id[row["Subcategory"]].append(row["ObjectId"])

# Calculate interesection over union for objects within subcategory per user.
excluded_subcats = []

user_iou_per_subcat = {}
for user, user_data in user_data_dict.items():
    iou_per_subcat = {}
    for subcat, object_id_list_subcat in subcategory_to_object_id.items():
        # print("****")
        placements_union_subcat = []
        placement_subsets_subcat = []
        
        for object_id in object_id_list_subcat:
            placements_per_object_id = []
            matching_rows = [
                i
                for i, x in enumerate(user_data)
                if x["object_id"] == object_id
            ]
            for row_num in matching_rows:
                for i in range(1, 4):
                    if (
                        f"room_{i}" in user_data[row_num]
                        and f"surface_{i}" in user_data[row_num]
                    ):
                        pl = (
                            user_data[row_num][f"room_{i}"],
                            user_data[row_num][f"surface_{i}"]
                        )
                        placements_per_object_id.append(pl)
                        if pl not in placements_union_subcat:
                            placements_union_subcat.append(pl)

            if not placements_per_object_id:
                if subcat not in excluded_subcats:
                    excluded_subcats.append(subcat)
                print(f"{user}/{subcat}/{object_id} has 0 placements.")
                continue
            # print(f"Object ID: {object_id}")
            # for p in placements_per_object_id: print(p)
            # print('----')
            placement_subsets_subcat.append(placements_per_object_id)

        # Calculate intersection over union on nested_list_placements.
        iou = 0
        if placements_union_subcat:
            for pl in placements_union_subcat:
                if all(pl in x for x in placement_subsets_subcat):
                    iou += 1
            iou /= len(placements_union_subcat)
        # print(f"User {user} subcategory {subcat}: iou {iou}")
        iou_per_subcat[subcat] = iou

    iou_per_subcat = dict(sorted(iou_per_subcat.items()))

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.bar(range(len(iou_per_subcat)), list(iou_per_subcat.values()), align='center')
    ax.set_xticks(range(len(iou_per_subcat)), list(iou_per_subcat.keys()), rotation=90)
    ax.set_title(f"User {user}")
    plt.savefig(f"../logs/data_statistics/{user}_iou_per_subcat.png")
    user_iou_per_subcat[user] = iou_per_subcat

# Filter excluded object ids and subcategories.
user_iou_per_subcat_filtered = {
    user: {
        subcat: user_iou_per_subcat[user][subcat]
        for subcat in user_iou_per_subcat[user]
        if subcat not in excluded_subcats
    }
    for user in user_iou_per_subcat
}

print([len(user_iou_per_subcat_filtered[user]) for user in user_iou_per_subcat_filtered])
assert all(
    len(user_iou_per_subcat_filtered[user]) == len(user_iou_per_subcat_filtered[list(user_iou_per_subcat_filtered.keys())[0]])
    for user in user_iou_per_subcat_filtered
)
# Find average iou per subcategory across users.
avg_iou_per_subcat = {}
std_iou_per_subcat = {}
for subcat in subcategory_to_object_id:
    if subcat in excluded_subcats:
        continue
    iou_per_users = [user_iou_per_subcat_filtered[user][subcat] for user in user_iou_per_subcat_filtered]
    avg_iou_per_subcat[subcat] = np.mean(iou_per_users)
    std_iou_per_subcat[subcat] = np.std(iou_per_users)

avg_iou_per_subcat = sorted(avg_iou_per_subcat.items(), key=lambda x: x[1])
avg_iou_per_subcat = {k: v for k, v in avg_iou_per_subcat}

for subcat, v in avg_iou_per_subcat.items():
    print(f"{subcat}: {v} +- {std_iou_per_subcat[subcat]}")
fig, ax = plt.subplots(figsize=(16, 16))
ax.bar(
    range(len(avg_iou_per_subcat)), list(avg_iou_per_subcat.values()),
    align='center', #yerr=[std_iou_per_subcat[x] for x in avg_iou_per_subcat]
)
ax.set_xticks(range(len(avg_iou_per_subcat)), list(avg_iou_per_subcat.keys()), rotation=90)
ax.set_title("Average IOU per subcategory across users")
plt.savefig("../logs/data_statistics/avg_iou_per_subcat.png")

with open("../object_data/memorization_subcats.txt", "w") as fwrite:
    for subcat in avg_iou_per_subcat:
        if avg_iou_per_subcat[subcat] <= 0.7:
            fwrite.write(f"{subcat}\n")

num_object_ids_excluded = 0
with open("../object_data/excluded_object_ids.txt", "w") as ftxt:
    for subcat, object_id_list in subcategory_to_object_id.items():
        if subcat in excluded_subcats:
            num_object_ids_excluded += len(object_id_list)
            for object_id in object_id_list:
                ftxt.write(f"{object_id}\n")
print(f"Number of subcategories excluded: {len(excluded_subcats)}")
print(f"Number of object ids excluded: {num_object_ids_excluded}")
