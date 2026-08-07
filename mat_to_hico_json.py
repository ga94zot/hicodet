"""Convert the official HICO-DET anno_bbox.mat annotations into the flat
per-image JSON format (file_name + hoi_annotation[*].category_id) expected
by llava/predict.py and analyze_hico.py.

category_id is the 1-indexed verb id (1-117) in alphabetical order of the
canonical HICO-DET verb name, matching llava/action_list.py's ordering.
"""
import json
import os

import numpy as np
import scipy.io as sio

HICODET_ROOT = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(HICODET_ROOT, "hico_20160224_det", "anno_bbox.mat")
OUT_DIR = os.path.join(HICODET_ROOT, "hico_20160224_det", "annotations")


def build_verb_id_map(list_action):
    vnames = sorted(set(x.vname for x in list_action))
    return {name: idx + 1 for idx, name in enumerate(vnames)}


def convert_split(bbox_entries, list_action, verb_id):
    images = []
    for entry in bbox_entries:
        hois = entry.hoi
        hois = [hois] if not isinstance(hois, np.ndarray) else hois
        hoi_annotation = []
        for hh in hois:
            if int(hh.invis) == 1:
                continue
            vname = list_action[hh.id - 1].vname
            category_id = verb_id[vname]
            num_instances = np.atleast_2d(hh.connection).shape[0]
            hoi_annotation.extend({"category_id": category_id} for _ in range(num_instances))
        images.append({"file_name": entry.filename, "hoi_annotation": hoi_annotation})
    return images


def main():
    mat = sio.loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)
    verb_id = build_verb_id_map(mat["list_action"])

    os.makedirs(OUT_DIR, exist_ok=True)
    trainval = convert_split(mat["bbox_train"], mat["list_action"], verb_id)
    test = convert_split(mat["bbox_test"], mat["list_action"], verb_id)

    with open(os.path.join(OUT_DIR, "trainval_hico.json"), "w") as f:
        json.dump(trainval, f)
    with open(os.path.join(OUT_DIR, "test_hico.json"), "w") as f:
        json.dump(test, f)

    print(f"trainval_hico.json: {len(trainval)} images")
    print(f"test_hico.json: {len(test)} images")


if __name__ == "__main__":
    main()
