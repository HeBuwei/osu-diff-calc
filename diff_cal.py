import json
import sys
from throughput import calculate_throughput

def cs_to_diameter(cs):
    # formula: (32.01*(1-(0.7*(cs-5)/5))) * 2
    return 108.834 - 8.9628 * cs

def calculate_distance(pos1, pos2):
    return ((pos1[0]-pos2[0]) ** 2 + (pos1[1]-pos2[1]) ** 2) ** 0.5

# Extract distance (D) and movement time (MT) between 2 hit objects
def extract_D_MT(prev_obj, curr_obj, diameter):
    
    if prev_obj['objectName'] == 'spinner' or curr_obj['objectName'] == 'spinner':
        D = 0.0
        MT = 1.0 # arbitrary
    
    elif prev_obj['objectName'] == 'slider':
        D = max(calculate_distance(prev_obj['endPosition'], curr_obj['position']) - 1.5 * diameter, 0.0)
        MT = (curr_obj['startTime'] - prev_obj['endTime'] + 30) / 1000.0
    
    elif prev_obj['objectName'] == 'circle':
        D = calculate_distance(prev_obj['position'], curr_obj['position'])
        MT = (curr_obj['startTime'] - prev_obj['startTime']) / 1000.0
    
    else: # huh?
        D = 0.0
        MT = 1.0       
    
    return (D, MT)


def calculate_map_diff(file_path):

    with open(file_path) as bm_file:
        
        bm = json.load(bm_file)

        hit_objects = bm['hitObjects']
        cs = float(bm['CircleSize'])
        diameter = cs_to_diameter(cs)

        Ds_MTs = []

        for prev_obj, curr_obj in zip(hit_objects, hit_objects[1:]):
            Ds_MTs.append(extract_D_MT(prev_obj, curr_obj, diameter))

        TP = calculate_throughput(Ds_MTs, diameter)

        return TP


if __name__ == "__main__":
    
    name = sys.argv[1]

    diff = calculate_map_diff('data/maps/' + name + '.json')

    print diff
