import json

with open("../data.json", "r") as jsonfile:
    all_user_data = json.load(jsonfile)

with open("../object_data/excluded_object_ids.txt", "r") as txtfile:
    excluded_object_ids = [l.strip() for l in txtfile.readlines()]

filtered_user_data = {}
for user_id, user_data in all_user_data.items():
    filtered_user_data[user_id] = []
    for object_data in user_data:
        object_id = object_data["object_id"]
        if object_id not in excluded_object_ids:
            filtered_user_data[user_id].append(object_data)
    print(f"User {user_id} has {len(filtered_user_data[user_id])} object annotations.")

with open("../data_filtered.json", "w") as jsonfile:
    json.dump(filtered_user_data, jsonfile, indent=4)