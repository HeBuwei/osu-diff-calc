import sys
import os
import json
from diff_cal import calculate_map_diff


if __name__ == "__main__":
    
    tag = sys.argv[1]

    if tag == "all":
        map_pack_to_compare = [x for x in os.listdir("data/maps") if x.endswith(".json")]
    else:
        map_packs = json.load(open("map_packs.json"))
        map_pack_to_compare = [x + ".json" for x in map_packs[tag]]

    results = []

    for filename in map_pack_to_compare:
        diff = calculate_map_diff("data/maps/" + filename)
        msg = '{:6.3f} - '.format(diff) + filename
        results.append((diff, msg))

    sorted_results = sorted(results, key=lambda tup:tup[0])

    for tup in sorted_results:
        print(tup[1])
