"""Concatenate CLIP and DINO embeddings for ablating the representation."""

import torch

clip_tensor_dict_bytype = {
    "max": torch.load("embeddings_saved/CLIP_max.pt"),
    "mean": torch.load("embeddings_saved/CLIP_mean.pt")
}

dino_tensor_dict_bytype = {
    "max": torch.load("embeddings_saved/DINO_max.pt"),
    "mean": torch.load("embeddings_saved/DINO_mean.pt")
}

for clip_type, clip_tensor_dict in clip_tensor_dict_bytype.items():
    for dino_type, dino_tensor_dict in dino_tensor_dict_bytype.items():
        new_representation_dict = {}
        for key in clip_tensor_dict:
            if key not in dino_tensor_dict:
                print(f"{key} not in dino_tensor_dict")
                continue
            assert clip_tensor_dict[key].shape[0] == 512 and dino_tensor_dict[key].shape[0] == 256
            new_representation_dict[key] = torch.concat(
                [clip_tensor_dict[key], dino_tensor_dict[key]], dim=0
            )
            assert new_representation_dict[key].shape[0] == 768
        torch.save(new_representation_dict, f"embeddings_saved/CLIP_{clip_type}_DINO_{dino_type}.pt")
