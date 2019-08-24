from collections import namedtuple
from math import sqrt

import numpy as np
from scipy import optimize
from scipy.special import expit
from scipy.linalg import norm
from scipy.interpolate import interp1d, CubicHermiteSpline
import matplotlib.pyplot as plt

from fitts_law import calc_IP, calc_hit_prob


P_THRESHOLD = 0.02

TP_MIN = 0.1
TP_MAX = 100


correction0_moving_spline = CubicHermiteSpline(np.array([-1,-0.6,0.3,0.5,1]),
                                               np.array([0.6,1,1,0.6,0]), 
                                               np.array([0.8,0.8,-0.8,-2,-0.8]))


# Data for correction calculation
# Refer to https://www.wolframcloud.com/obj/hebuweitom/Published/Correction.nb
# for the effects of the data in graphs
# obj0 - flow
a0f = [0.5, 1, 1.5, 2]
k0f = interp1d(a0f, [-5.5,-7.5,-10,-8.2], bounds_error=False, fill_value=(-5.5,-8.2))

coeffs0f = np.array([[[-0.5,0,1,5],
                      [-0.35,0.35,1,0],
                      [-0.35,-0.35,1,0]],
                     [[-1,0,1,4],
                      [-0.7,0.7,1,1],
                      [-0.7,-0.7,1,1]],
                     [[-1.5,0,1,2],
                      [-1,1,1,2],
                      [-1,-1,1,2]],
                     [[-2,0,1,2],
                      [-1.4,1.4,1,2],
                      [-1.4,-1.4,1,2]]])

components0f = interp1d(a0f, coeffs0f, axis=0, bounds_error=False, fill_value=(coeffs0f[0], coeffs0f[-1]))

# obj0 - snap
a0s = [1.5, 2.5, 4, 6]
k0s = interp1d(a0s, [-1,-5.9,-5,-3.7], bounds_error=False, fill_value=(-1,-3.7))

coeffs0s = np.array([[[2,0,1,1],
                      [1.6,2,1,0],
                      [1.6,2,1,0],
                      [0,0,1,0]],
                     [[3,0,1,1],
                      [1.8,2.4,1,0.3],
                      [1.8,-2.4,1,0.3],
                      [0,0,1,-0.3]],
                     [[4,0,0,0.6],
                      [2,4,1,0.24],
                      [2,-4,1,0.24],
                      [-1,0,1,-0.24]],
                     [[6,0,0,0.4],
                      [3,6,1,0.16],
                      [3,-6,1,0.16],
                      [-1.5,0,1,-0.16]]])

components0s = interp1d(a0s, coeffs0s, axis=0, bounds_error=False, 
                      fill_value=(coeffs0s[0], coeffs0s[-1]))

# obj3 - flow
a3f = [0, 1, 2, 3]
k3f = interp1d(a3f, [-4,-4,-4.5,-2.5], bounds_error=False, fill_value=(-4,-2.5))

coeffs3f = np.array([[[0,0,0,1.5],
                      [0,0,0,2]],
                     [[1,0,0,1.5],
                      [0,0,0,2]],
                     [[2,0,0,1],
                      [0,0,0,2.5]],
                     [[4,0,0,0],
                      [0,0,0,3.5]]])

components3f = interp1d(a3f, coeffs3f, axis=0, bounds_error=False, 
                      fill_value=(coeffs3f[0], coeffs3f[-1]))

# obj3 - snap
a3s = [1.5, 2.5, 4, 6]
k3s = interp1d(a3s, [-1.8,-3,-5.4,-4.9], bounds_error=False, fill_value=(-1.8,-4.9))

coeffs3s = np.array([[[-2,0,1,0.4],
                      [-1,1.4,1,0],
                      [-1,-1.4,1,0],
                      [0,0,0,2],
                      [1,0,1,-1]],
                     [[-3,0,1,0.2],
                      [-1.5,2.1,1,0.2],
                      [-1.5,-2.1,1,0.2],
                      [0,0,0,1],
                      [1.5,0,1,-0.6]],
                     [[-4,0,0,0.4],
                      [-2,2,1,0.4],
                      [-2,-2,1,0.4],
                      [0,0,0,0.6],
                      [2,0,1,-0.4]],
                     [[-6,0,0,0.3],
                      [-3,3,1,0.2],
                      [-3,-3,1,0.2],
                      [0,0,0,0.6],
                      [3,0,1,-0.3]]])

components3s = interp1d(a3s, coeffs3s, axis=0, bounds_error=False, 
                      fill_value=(coeffs3s[0], coeffs3s[-1]))


Movement = namedtuple('Movement', ['D', 'MT', 'time', 'D_raw', 'D_corr0', 'aim_strain'])



def calc_aim_diff(beatmap, analysis=False):

    hit_objects = beatmap['hitObjects']
    diameter = cs_to_diameter(beatmap["CsAfterMods"])

    movements = extract_movements(hit_objects, diameter)
    
    adjusted_movements = calc_adjusted_movements(movements, diameter)

    TP = calc_throughput(adjusted_movements, diameter)
    diff = TP ** 0.85 * 0.585

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
    
    # if obj1['objectName'] == 'slider':

    #     finish_position = get_finish_position(obj1)

    #     # This is only a temporary algorithm to sliders
    #     # long sliders (when the slider tail matters)
    #     D_long = max(calc_distance(finish_position, obj2['position']) - 1.5 * diameter, 0.0)
    #     MT_long = (obj2['startTime'] - obj1['endTime'] + 70) / 1000.0

    #     # short sliders (when the slider head matters) (treat as a circle)
    #     D_short = calc_distance(obj1['position'], obj2['position'])
    #     MT_short = (obj2['startTime'] - obj1['startTime']) / 1000.0

    #     if calc_IP(D_long, diameter, MT_long) > calc_IP(D_short, diameter, MT_short):
    #         D = D_long
    #         MT = MT_long
    #     else:
    #         D = D_short
    #         MT = MT_short

    # elif obj1['objectName'] == 'circle':

    #     finish_position = obj1['position']

    #     D = calc_distance(obj1['position'], obj2['position'])
    #     MT = (obj2['startTime'] - obj1['startTime']) / 1000.0
    
    # else: 
    #     raise Exception  

    finish_position = obj1['position']
    D = calc_distance(obj1['position'], obj2['position'])
    MT = (obj2['startTime'] - obj1['startTime']) / 1000.0
    

    IP = calc_IP(D, diameter, MT)

    obj1_in_the_middle = False
    obj2_in_the_middle = False

    s12 = (np.array(obj2['position']) - np.array(finish_position)) / diameter
    d12 = norm(s12)


    # Correction #1 - The Previous Object
    # Estimate how obj0 affects the difficulty of hitting obj2
    correction0 = 0

    if obj0 is not None:
        
        s01 = (np.array(obj1['position']) - np.array(obj0['position'])) / diameter
        
        if d12 == 0:
            correction0 = 0

        else:
            d01 = norm(s01)
            t01 = (obj1['startTime'] - obj0['startTime']) / 1000.0
            t12 = MT
            t_ratio = t12 / t01

            if t_ratio > 1.4:
                # s01*t_ratio, s12

                if d01 == 0:
                    correction0 = 0.2
                    pass

                else:
                    cos012 = np.clip(-s01.dot(s12) / d01 / d12, -1, 1)

                    correction_moving = correction0_moving_spline(cos012) * 1.0
                    correction_still = 0.2

                    movingness = expit(d01 * 2) * 2 - 1
                    correction0 = (movingness * correction_moving + (1-movingness) * correction_still) * 0.8

            elif t_ratio < 1/1.4:

                if d01 == 0:
                    correction0 = 0

                else:
                    cos012 = np.clip(-s01.dot(s12) / d01 / d12, -1, 1)
                    correction0 = (1 - cos012) * expit((d01*t_ratio - 1.5) * 4) * 0.3

            else:

                obj1_in_the_middle = True
                normalized_pos0 = -s01 / t01 * t12
                x0 = normalized_pos0.dot(s12) / d12
                y0 = norm(normalized_pos0 - x0 * s12 / d12)

                correction0_flow = calc_correction0_flow(d12, x0, y0)
                correction0_snap = calc_correction0_snap(d12, x0, y0)

                i = -10
                correction0 = ((correction0_flow**i + correction0_snap**i) / 2) ** (1/i)

                # print('{:8} {:16.13f}'.format(obj2['startTime'], correction0))

    # print('{:8} {:16.13f} {:16.13f}'.format(obj2['startTime'], d12, D))

    # Correction #2 - The Next Object
    # Estimate how obj3 affects the difficulty of hitting obj2
    correction3 = 0

    if obj3 is not None:

        s23 = (np.array(obj3['position']) - np.array(obj2['position'])) / diameter

        if d12 == 0:
            correction3 = 0

        else:
            d23 = norm(s23)
            t12 = MT
            t23 = (obj3['startTime'] - obj2['startTime']) / 1000.0
            t_ratio = t12 / t23

            if t_ratio > 1.4:

                if d23 == 0:
                    correction3 = 0

                else:
                    cos123 = np.clip(-s12.dot(s23) / d12 / d23, -1, 1)

                    correction_moving = correction0_moving_spline(cos123) * 1.0
                    correction_still = 0

                    movingness = expit(d23 * 6 - 5) - expit(-5)
                    correction3 = (movingness * correction_moving + (1-movingness) * correction_still) * 0.5

            elif t_ratio < 1/1.4:

                if d23 == 0:
                    correction3 = 0

                else:
                    cos123 = np.clip(-s12.dot(s23) / d12 / d23, -1, 1)
                    correction3 = (1 - cos123) * expit((d23*t_ratio - 1.5) * 4) * 0.15

            else:

                obj2_in_the_middle = True
                normalized_pos3 = s23 / t23 * t12
                x0 = normalized_pos3.dot(s12) / d12
                y0 = norm(normalized_pos3 - x0 * s12 / d12)

                correction3_flow = calc_correction3_flow(d12, x0, y0)
                correction3_snap = calc_correction3_snap(d12, x0, y0)

                i = -10
                correction3 = max(((correction3_flow**i + correction3_snap**i) / 2) ** (1/i) - 0.1, 0) * 0.5

                # print('{:8} {:16.13f}'.format(obj2['startTime'], correction0))



    # Correction #3 - 4-object pattern
    # Estimate how the whole pattern consisting of obj0 to obj3 affects 
    # the difficulty of hitting obj2. This only takes effect when the pattern
    # is not so spaced (i.e. does not contain jumps) 
    correction_pattern = 0

    if obj1_in_the_middle and obj2_in_the_middle:

       gap = norm(s12 - s23/2 - s01/2)
       spacing = np.prod([((d**10 + 1) / 2) ** (1/10) for d in [d01, d12, d23]])

       correction_pattern = (expit((gap-0.75)*8) - expit(-6)) * (1 - expit((spacing-3)*4)) * 0.4


    # Correction #4 - Tap Strain
    # Estimate how tap strain affects difficulty
    correction_tap = 0

    if 'tapStrain' in obj2 and D > 0:
        tap_strain = obj2['tapStrain']
        correction_tap = expit((np.average(tap_strain) / IP - 1) * 15) * 0.2



    # Correction #5 - Cheesing
    # The player might make the movement of obj1 -> obj2 easier by 
    # hitting obj1 early and obj2 late. Here we estimate the amount of 
    # cheesing and update MT accordingly.

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

        correction_early = expit((IP01 / IP - 0.6) * (-15)) * (1 / (1/(MT+0.07) + MT01_recp)) * 0.15

        if obj3 is not None:
            D23 = calc_distance(obj2['position'], obj3['position'])
            MT23 = (obj3['startTime'] - obj2['startTime']) / 1000.0
            MT23_recp = 1 / MT23
            IP23 = calc_IP(D23, diameter, MT23)
        else:
            MT23_recp = 0
            IP23 = 0

        correction_late = expit((IP23/IP - 0.6) * (-15)) * (1 / (1/(MT+0.07) + MT23_recp)) * 0.15


    # Correction #6 - Small circle bonus
    small_circle_bonus = expit((55 - diameter) / 2) * 0.1


    # apply the corrections above
    D_raw = D * (1 + small_circle_bonus)
    D_corr0 = D_raw * (1 + correction0 + correction3 + correction_pattern)
    D_corr_tap = D_corr0 * (1 + correction_tap)


    MT += correction_early + correction_late

    return Movement(D_corr_tap, MT, obj2['startTime'] / 1000, D_raw, D_corr0, 0)


# Refer to https://www.wolframcloud.com/obj/hebuweitom/Published/Correction.nb
# for the effects of the 4 functions below in graphs
def calc_correction0_flow(d, x0, y0):

    correction_raw = k0f(d)

    for c in components0f(d):
        correction_raw += c[3] * sqrt((x0-c[0])**2 + (y0-c[1])**2 + c[2])

    return expit(correction_raw)


def calc_correction0_snap(d, x0, y0):

    correction_raw = k0s(d)

    for c in components0s(d):
        correction_raw += c[3] * sqrt((x0-c[0])**2 + (y0-c[1])**2 + c[2])

    return expit(correction_raw)


def calc_correction3_flow(d, x0, y0):

    correction_raw = k3f(d)

    for c in components3f(d):
        correction_raw += c[3] * sqrt((x0-c[0])**2 + (y0-c[1])**2 + c[2])

    return expit(correction_raw)


def calc_correction3_snap(d, x0, y0):

    correction_raw = k3s(d)

    for c in components3s(d):
        correction_raw += c[3] * sqrt((x0-c[0])**2 + (y0-c[1])**2 + c[2])

    return expit(correction_raw)


# Returns a new list of movements, of which the MT is adjusted based on aim strain
# Currently NOT IN USE
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
    # formula: (1-(0.7*(cs-5)/5)) * 32 * 2
    return 108.8 - 8.96 * cs


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

