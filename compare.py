import csv
import sys
import os
import json
from time import localtime, strftime
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

        results.append(['{:6.3f}'.format(d) for d in list(diffs)] + 
                       [map_n_mods[1], beatmap["Title"] + " [" + beatmap["Version"] + "]", beatmap_id])

    sorted_results = sorted(results, key=lambda tup:tup[0])
    # sorted_results = sorted(results, key=lambda tup:tup[1])

    time_str = strftime("%H%M%S", localtime())

    with open('data/results/{}.csv'.format(time_str), 'w', newline='') as csvfile:
        cw = csv.writer(csvfile)
        cw.writerow([" Aim  ", " Tap  ", " Overall", "Mods", "Beatmap", "ID"])
        for result in sorted_results:
            cw.writerow(result)
