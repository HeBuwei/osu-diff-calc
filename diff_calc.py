import json
import sys

import numpy as np

from fitts_law import calc_throughput, calc_IP


def cs_to_diameter(cs):
    # formula: (32.01*(1-(0.7*(cs-5)/5))) * 2
    return 108.834 - 8.9628 * cs


def calc_distance(pos1, pos2):
    return ((pos1[0]-pos2[0]) ** 2 + (pos1[1]-pos2[1]) ** 2) ** 0.5


def get_finish_position(slider):
    if slider['repeatCount'] % 2 == 0:
        return slider['position']
    else:
        return slider['endPosition']


# Extract distance (D) and movement time (MT) between 2 hit objects
def extract_D_MT(diameter, obj1, obj2):
    
    if obj1['objectName'] == 'slider':

        finishPosition = get_finish_position(obj1)

        # long sliders (when the slider tail matters)
        D_long = max(calc_distance(finishPosition, obj2['position']) - 1.5 * diameter, 0.0)
        MT_long = (obj2['startTime'] - obj1['endTime'] + 70) / 1000.0

        # short sliders (when the slider head matters) (treat as a circle)
        D_short = calc_distance(obj1['position'], obj2['position'])
        MT_short = (obj2['startTime'] - obj1['startTime']) / 1000.0

        if calc_IP(D_long, diameter, MT_long) > calc_IP(D_short, diameter, MT_short):
            D = D_long
            MT = MT_long
        else:
            D = D_short
            MT = MT_short

    elif obj1['objectName'] == 'circle':
        D = calc_distance(obj1['position'], obj2['position'])
        MT = (obj2['startTime'] - obj1['startTime']) / 1000.0
    
    else:
        raise Exception
    
    return (D, MT)


def speed_up(hit_objects, factor):

    for hit_object in hit_objects:

        hit_object["startTime"] = hit_object["startTime"] / factor

        if "endTime" in hit_object:
            hit_object["endTime"] = hit_object["endTime"] / factor


def apply_mods(beatmap, mods):

    hit_objects = beatmap['hitObjects']
    cs = float(beatmap['CircleSize'])

    if mods[0] == "hr":
        cs = 1.3 * cs
    elif mods[0] == "ez":
        cs = 0.5 * cs

    beatmap["CsAfterMods"] = cs

    if mods[1] == "dt":
        speed_up(hit_objects, 1.5)
    elif mods[1] == "ht":
        speed_up(hit_objects, 0.75)


def remove_spinners(beatmap):
    beatmap['hitObjects'] = [obj for obj in beatmap['hitObjects'] if obj['objectName'] != 'spinner']


def calc_diff(beatmap, mods=["nm", "nm"]):
    
    remove_spinners(beatmap)
    apply_mods(beatmap, mods)
    
    aim_diff = calc_aim_diff_corrected(beatmap)
    return aim_diff


def calc_aim_diff_naive(beatmap):

    hit_objects = beatmap['hitObjects']
    diameter = cs_to_diameter(beatmap["CsAfterMods"])

    Ds_MTs = []

    for obj1, obj2 in zip(hit_objects, hit_objects[1:]):
        Ds_MTs.append(extract_D_MT(diameter, obj1, obj2))

    TP = calc_throughput(Ds_MTs, diameter)
    
    diff = TP / 2.5
    return diff


def calc_aim_diff_corrected(beatmap):

    hit_objects = beatmap['hitObjects']
    diameter = cs_to_diameter(beatmap["CsAfterMods"])

    Ds_MTs = [extract_D_MT_corrected(diameter, hit_objects[0], hit_objects[1], obj3=hit_objects[2])]

    for obj0, obj1, obj2, obj3 in zip(hit_objects, hit_objects[1:], hit_objects[2:], hit_objects[3:]):
        Ds_MTs.append(extract_D_MT_corrected(diameter, obj1, obj2, obj0=obj0, obj3=obj3))

    Ds_MTs.append(extract_D_MT_corrected(diameter, hit_objects[-2], hit_objects[-1], obj0=hit_objects[-3]))

    TP = calc_throughput(Ds_MTs, diameter)

    diff = TP / 2.6
    return diff


# Extract D and MT of the movement from obj1 and obj2 and adjust the values
# by taking the neighbouring objects into consideration
def extract_D_MT_corrected(diameter, obj1, obj2, obj0=None, obj3=None):
    
    if obj1['objectName'] == 'slider':

        finish_position = get_finish_position(obj1)

        # long sliders (when the slider tail matters)
        D_long = max(calc_distance(finish_position, obj2['position']) - 1.5 * diameter, 0.0)
        MT_long = (obj2['startTime'] - obj1['endTime'] + 70) / 1000.0

        # short sliders (when the slider head matters) (treat as a circle)
        D_short = calc_distance(obj1['position'], obj2['position'])
        MT_short = (obj2['startTime'] - obj1['startTime']) / 1000.0

        if calc_IP(D_long, diameter, MT_long) > calc_IP(D_short, diameter, MT_short):
            D = D_long
            MT = MT_long
        else:
            D = D_short
            MT = MT_short

    elif obj1['objectName'] == 'circle':

        finish_position = obj1['position']

        D = calc_distance(obj1['position'], obj2['position'])
        MT = (obj2['startTime'] - obj1['startTime']) / 1000.0
    
    else: 
        raise Exception

    if obj3 is not None:

        v1 = np.array(obj2['position']) - np.array(finish_position)
        v2 = np.array(obj3['position']) - np.array(obj2['position'])

        if np.sqrt(v1.dot(v1)) == 0:
            correction_factor = 0
        else:
            correction_factor = min((v2.dot(v1) / np.sqrt(v1.dot(v1)) + np.sqrt(v2.dot(v2))) / diameter * 0.05, 0.2)
        
        t2 = (obj3['startTime'] - obj2['startTime']) / 1000.0

        correction_factor = correction_factor * max(1 - 2.5 * t2, 0)
        D = D * (1 + correction_factor)
    
    return (D, MT)


def calc_file_diff(file_path, mods=["nm", "nm"]):
    bm = load_beatmap(file_path)
    return calc_diff(bm, mods)


def calc_IP_vs_time(file_path):
        
    bm = load_beatmap(file_path)

    hit_objects = bm['hitObjects']
    cs = float(bm['CircleSize'])
    diameter = cs_to_diameter(cs)

    Ds_MTs = []
    times = []

    for obj1, obj2 in zip(hit_objects, hit_objects[1:]):
        Ds_MTs.append(extract_D_MT(diameter, obj1, obj2))
        times.append(obj2['startTime'])

    IPs = [calc_IP(D, diameter, MT) for (D, MT) in Ds_MTs]

    return (IPs, times)


def load_beatmap(file_path):
    with open(file_path, encoding="utf8") as bm_file:
        bm = json.load(bm_file)
        return bm


if __name__ == "__main__":
    
    name = sys.argv[1]
    mods = [sys.argv[2], sys.argv[3]]

    diff = calc_file_diff('data/maps/' + name + '.json', mods)

    print(diff)