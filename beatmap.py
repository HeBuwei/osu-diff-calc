import json


def load_beatmap(file_path):
    with open(file_path, encoding="utf8") as bm_file:
        bm = json.load(bm_file)
        return bm
