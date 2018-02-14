import os
from diff_cal import calculate_map_diff

for filename in os.listdir("data/maps"):
    if filename.endswith(".json"):
        diff = calculate_map_diff("data/maps/" + filename)
        print('{:6.3f} - '.format(diff) + filename)
        