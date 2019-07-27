import sys
import os
import json
from diff_calc import calc_diff
from mods import str_to_mods
from beatmap import load_beatmap


if __name__ == "__main__":
    
    tag = sys.argv[1]

    if tag == "all":
        map_pack = [[x.rsplit('.', maxsplit=1)[0], '--']
                    for x in os.listdir("data/maps")
                    if x.endswith(".json")]

    else:
        map_packs = json.load(open("map_packs.json"))
        map_pack = map_packs[tag]

    results = []

    for map_n_mods in map_pack:
        map_file_name = map_n_mods[0]
        beatmap = load_beatmap("data/maps/" + map_file_name + ".json")
        diffs = calc_diff(beatmap, mods=str_to_mods(map_n_mods[1]))
        # diffs = calc_diff(beatmap, mods=str_to_mods(map_n_mods[1]), tap_details=True)

        if "BeatmapID" in beatmap:
            beatmap_id = beatmap["BeatmapID"]
        else:
            beatmap_id = ""

        # map_name = beatmap_id + " " + beatmap["Title"] + " [" + beatmap["Version"] + "]"
        map_name = "{:.40} [{:.20}] {}".format(beatmap["Title"], beatmap["Version"], beatmap_id)

        msg = ''.join(['{:6.3f} | '.format(diff) for diff in diffs])
        msg += map_n_mods[1] + " " + map_name
        results.append(diffs + (msg,))

    sorted_results = sorted(results, key=lambda tup:tup[0])
    # sorted_results = sorted(results, key=lambda tup:tup[1])
    # sorted_results = sorted(results, key=lambda tup:tup[-1])

    print(" Aim      Tap     Overall")


    for tup in sorted_results:
        print(tup[-1])
