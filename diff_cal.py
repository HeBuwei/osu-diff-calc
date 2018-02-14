import json
import sys
from fitts_law import calculate_throughput, calculate_IP


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
        MT = (curr_obj['startTime'] - prev_obj['endTime'] + 70) / 1000.0 # TODO: very short slider
    
    elif prev_obj['objectName'] == 'circle':
        D = calculate_distance(prev_obj['position'], curr_obj['position'])
        MT = (curr_obj['startTime'] - prev_obj['startTime']) / 1000.0
    
    else: # huh?
        D = 0.0
        MT = 1.0       
    
    return (D, MT)


def calculate_map_diff(file_path):
        
    bm = load_beatmap(file_path)

    hit_objects = bm['hitObjects']
    cs = float(bm['CircleSize'])
    diameter = cs_to_diameter(cs)

    Ds_MTs = []

    for prev_obj, curr_obj in zip(hit_objects, hit_objects[1:]):
        Ds_MTs.append(extract_D_MT(prev_obj, curr_obj, diameter))

    TP = calculate_throughput(Ds_MTs, diameter)

    diff = TP / 2.5

    return diff


def calculate_IP_vs_time(file_path):
        
    bm = load_beatmap(file_path)

    hit_objects = bm['hitObjects']
    cs = float(bm['CircleSize'])
    diameter = cs_to_diameter(cs)

    Ds_MTs = []
    times = []

    for prev_obj, curr_obj in zip(hit_objects, hit_objects[1:]):
        Ds_MTs.append(extract_D_MT(prev_obj, curr_obj, diameter))
        times.append(curr_obj['startTime'])

    IPs = [calculate_IP(D, diameter, MT) for (D, MT) in Ds_MTs]

    return (IPs, times)



def load_beatmap(file_path):
    with open(file_path, encoding="utf8") as bm_file:
        bm = json.load(bm_file)
        return bm


if __name__ == "__main__":
    
    name = sys.argv[1]

    diff = calculate_map_diff('data/maps/' + name + '.json')

    print(diff)
