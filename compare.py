import sys
import os
import json
from diff_calc import calc_file_diff
from mods import str_to_mods


if __name__ == "__main__":
    
    tag = sys.argv[1]

    if tag == "all":
        map_pack = [[x.rsplit('.', maxsplit=1)[0], '-']
                    for x in os.listdir("data/maps")
                    if x.endswith(".json")]

    else:
        map_packs = json.load(open("map_packs.json"))
        map_pack = map_packs[tag]

    results = []

    for map_n_mods in map_pack:
        songname = map_n_mods[0]
        diffs = calc_file_diff("data/maps/" + songname + ".json", mods=str_to_mods(map_n_mods[1]))
        msg = ''.join(['{:6.3f} | '.format(diff) for diff in diffs])
        msg += songname + " " + map_n_mods[1]
        results.append(diffs + (msg,))

    sorted_results = sorted(results, key=lambda tup:tup[0])

    for tup in sorted_results:
        print(tup[-1])
