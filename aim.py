from collections import namedtuple

import numpy as np
from scipy import optimize
from scipy.special import expit
from scipy.linalg import norm
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt


from fitts_law import calc_IP, calc_hit_prob


P_THRESHOLD = 0.02

TP_MIN = 0.1
TP_MAX = 100


correction0_moving_spline = CubicHermiteSpline(np.array([-1,-0.5,-0.3,0.6,1]),
                                               np.array([0,0.6,1,1,0.6]), 
                                               np.array([0.8,2,0.8,-0.8,-0.8]))


Movement = namedtuple('Movement', ['D', 'MT', 'time', 'D_raw', 'D_corr0', 'aim_strain'])



def calc_aim_diff(beatmap, analysis=False):

    hit_objects = beatmap['hitObjects']
    diameter = cs_to_diameter(beatmap["CsAfterMods"])

    movements = extract_movements(hit_objects, diameter)
    
    adjusted_movements = calc_adjusted_movements(movements, diameter)

    TP = calc_throughput(adjusted_movements, diameter)
    diff = TP ** 0.85 * 0.618

    if analysis:
        return (TP, adjusted_movements)
    else:
        return diff


def extract_movements(hit_objects, diameter):

    movements = []

    if len(hit_objects) == 2:

        movements.append(extract_movement(diameter, hit_objects[0], hit_objects[1]))

    elif len(hit_objects) >= 3:

        movements.append(extract_movement(diameter, hit_objects[0], hit_objects[1], obj3=hit_objects[2]))

        for obj0, obj1, obj2, obj3 in zip(hit_objects, hit_objects[1:], hit_objects[2:], hit_objects[3:]):
            movements.append(extract_movement(diameter, obj1, obj2, obj0=obj0, obj3=obj3))

        movements.append(extract_movement(diameter, hit_objects[-2], hit_objects[-1], obj0=hit_objects[-3]))

    return movements


# Extract information about the movement from obj1 and obj2
# including D, MT, end time of the movement, and the factors of correction
def extract_movement(diameter, obj1, obj2, obj0=None, obj3=None):
    
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

    
    IP = calc_IP(D, diameter, MT)


    # Correction #1 - The Previous Object
    # Estimate how obj0 affects the difficulty of obj1 -> obj2 and make correction accordingly
    # Very empirical, may need tweaking
    # correction_snap = 0
    # correction_flow = 0

    correction0 = 0

    if obj0 is not None:
        
        s01 = (np.array(obj1['position']) - np.array(obj0['position'])) / diameter
        s12 = (np.array(obj2['position']) - np.array(finish_position)) / diameter
        
        if norm(s12) == 0:
            # correction_lvl_snap = 0
            # correction_lvl_flow = 0
            # correction_snap = 0
            correction0 = 0


        else:
            t01 = (obj1['startTime'] - obj0['startTime']) / 1000.0
            t12 = MT

            v01 = s01 / t01
            v12 = s12 / t12
            dv = v12 - v01

            a01 = -4 * s01 / t01 ** 2
            a12 = 4 * s12 / t12 ** 2
            da = a12 - a01


            # Version 1
            # correction_lvl_flow = dv.dot(dv) / (2 * (v12.dot(v12) + v01.dot(v01)))
            # correction_lvl_snap = da.dot(da) / (2 * (a12.dot(a12) + a01.dot(a01)))

            # flowiness = expit((norm(s12) - 1.45) * (-10))
            # snappiness = expit((norm(s12) - 1.4) * 10)

            # correction_flow = 0.5 ** (t01 / t12 / 2) * correction_lvl_flow * flowiness * 1
            # correction_snap = 0.5 ** (t01 / t12 / 2) * correction_lvl_snap * snappiness * 1


            # Version 2

            t_ratio = t12 / t01

            if t_ratio > 1.7:
                # s01*t_ratio, s12

                if norm(s01) == 0:
                    correction0 = 0.2

                else:
                    cos_012 = np.clip(s01.dot(s12) / norm(s01) / norm(s12), -1, 1)

                    correction_moving = correction0_moving_spline(cos_012) * 1.0

                    correction_still = 0.2

                    movingness = expit(norm(s01) * 2) * 2 - 1
                    correction0 = movingness * correction_moving + (1-movingness) * correction_still


            elif t_ratio < 1/1.7:
                pass
            else:
                pass





    # Correction #2 - The Next Object
    # Estimate how obj3 affects the difficulty of obj1 -> obj2 and make correction accordingly
    # Again, very empirical
    correction_obj3 = 0

    # if obj3 is not None:

    #     s1 = (np.array(obj1['position']) - np.array(obj2['position'])) / diameter
    #     s3 = (np.array(obj3['position']) - np.array(obj2['position'])) / diameter

    #     # t1 = -1
    #     # t3 = (obj3['startTime'] - obj2['startTime']) / (obj2['startTime'] - obj1['startTime'])

    #     t1 = (obj1['startTime'] - obj2['startTime']) / 1000
    #     t3 = (obj3['startTime'] - obj2['startTime']) / 1000

    #     x_params = np.linalg.solve(np.array([[t1*t1/2, t1],
    #                                          [t3*t3/2, t3]]),
    #                                np.array([s1[0], s3[0]]))

    #     y_params = np.linalg.solve(np.array([[t1*t1/2, t1],
    #                                          [t3*t3/2, t3]]),
    #                                np.array([s1[1], s3[1]]))

    #     a = np.array([x_params[0], y_params[0]])
    #     v = np.array([x_params[1], y_params[1]])


    #     print(obj2['startTime'], ", ",
    #           ' | '.join(['{:6.3f}'.format(x) for x in [norm(v), norm(v)/(IP+0.1), v[0], v[1], a[0], a[1]]]))
        # print(a, ", ", v)



    # Correction #3 - Tap Strain
    # Estimate how tap strain affects difficulty
    correction_tap = 0

    if 'tapStrain' in obj2 and D > 0:
        tap_strain = obj2['tapStrain']
        correction_tap = expit((np.average(tap_strain) / IP - 1) * 15) * 0.2



    # Correction #4 - Cheesing
    # The player might make the movement of obj1 -> obj2 easier by 
    # hitting obj1 early and obj2 late. 

    correction_early = correction_late = 0

    if D > 0:

        if obj0 is not None:
            D01 = calc_distance(obj0['position'], obj1['position'])
            MT01 = (obj1['startTime'] - obj0['startTime']) / 1000.0
            MT01_recp = 1 / MT01
            IP01 = calc_IP(D01, diameter, MT01)
        else:
            MT01_recp = 0
            IP01 = 0

        correction_early = expit((IP01 / IP - 0.6) * (-15)) * (1 / (1/(MT+0.07) + MT01_recp)) * 0.12

        if obj3 is not None:
            D23 = calc_distance(obj2['position'], obj3['position'])
            MT23 = (obj3['startTime'] - obj2['startTime']) / 1000.0
            MT23_recp = 1 / MT23
            IP23 = calc_IP(D23, diameter, MT23)
        else:
            MT23_recp = 0
            IP23 = 0

        correction_late = expit((IP23/IP - 0.6) * (-15)) * (1 / (1/(MT+0.07) + MT23_recp)) * 0.12


    # apply the corrections

    D_raw = D
    # MT -= 0.05 * correction_snap
    # D *= 1 + correction_snap
    
    # D_corr0 = D_raw
    D_corr0 = D_raw + correction0 * D_raw * 0.5 + correction0 * diameter * 0.5 

    D_corr_tap = D_corr0 * (1 + correction_tap)
    # D_corr_tap = D_corr0

    MT += correction_early + correction_late

    return Movement(D_corr_tap, MT, obj2['startTime'] / 1000, D_raw, D_corr0, 0)




# Returns a new list of movements, of which the MT is adjusted based on aim strain
def calc_adjusted_movements(movements, diameter):

    adjusted_movements = []
    curr_strain = 0
    prev_time = 0.0
    k = 2

    for mvmt in movements:

        curr_time = mvmt.time
        curr_strain *= np.exp(-k * (curr_time - prev_time))

        IP_old = calc_IP(mvmt.D, diameter, mvmt.MT)

        if IP_old == 0:
            adjustment = 0
        else:
            adjustment = expit((curr_strain / IP_old - 0.7) * 15) * 0.1 - 0.05

        MT_new = mvmt.MT / (1 + adjustment)

        adjusted_movements.append(mvmt._replace(aim_strain=curr_strain))
        # adjusted_movements.append(mvmt._replace(MT=MT_new, aim_strain=curr_strain))

        curr_strain += np.log2(mvmt.D / diameter + 1) * k
        prev_time = curr_time

    return adjusted_movements



# calculates the throughput required to fc the map with probability P_THRESHOLD
def calc_throughput(movements, W):

    fc_prob_TP_min = calc_fc_prob(TP_MIN, movements, W)
    
    # if map is so easy that players with minimum throughput can fc with decent possibility
    if fc_prob_TP_min >= P_THRESHOLD:
        return TP_MIN

    fc_prob_TP_max = calc_fc_prob(TP_MAX, movements, W)

    # if map is too hard 
    if fc_prob_TP_max <= P_THRESHOLD:
        return TP_MAX

    # x, r = optimize.brentq(calc_fc_prob_minus_threshold, TP_MIN, TP_MAX, args=(movements, W), full_output=True)
    # print(r.iterations)

    x = optimize.brentq(calc_fc_prob_minus_threshold, TP_MIN, TP_MAX, args=(movements, W))

    return x


def calc_fc_prob_minus_threshold(TP, movements, W):
    return calc_fc_prob(TP, movements, W) - P_THRESHOLD
    

def calc_fc_prob(TP, movements, W):
    fc_prob = 1.0

    for mvmt in movements:
        D = mvmt[0]
        MT = mvmt[1]
        hit_prob = calc_hit_prob(D, W, MT, TP)
        fc_prob *= hit_prob

    # print('' + str(TP) + ' | ' + str(fc_prob))
    return fc_prob




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


if __name__ == '__main__':
    xnew = np.linspace(-1, 1, num=201, endpoint=True)
    plt.plot(xnew, correction0_moving_spline(xnew), '-')
    plt.xlabel('cosine of angle 0-1-2')
    plt.ylabel('correction factor')
    plt.legend(['spline'], loc='best')
    plt.show()
