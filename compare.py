import sys
import os
import json
from diff_calc import calc_file_diff


if __name__ == "__main__":
    
    tag = sys.argv[1]

    if tag == "all":
        map_pack_to_compare = [x for x in os.listdir("data/maps") if x.endswith(".json")]
    else:
        map_packs = json.load(open("map_packs.json"))
        map_pack_to_compare = [x + ".json" for x in map_packs[tag]]

    results = []

    for filename in map_pack_to_compare:
        songname = filename.rsplit('.', maxsplit=1)[0]
        aim_diff, tap_diff, overall_diff = calc_file_diff("data/maps/" + filename)
        msg = ('{:6.3f} - '.format(aim_diff) 
               + '{:6.3f} - '.format(tap_diff) 
               + '{:6.3f} - '.format(overall_diff) 
               + songname)
        results.append((aim_diff, tap_diff, overall_diff, msg))

    sorted_results = sorted(results, key=lambda tup:tup[0])

    for tup in sorted_results:
        print(tup[3])
