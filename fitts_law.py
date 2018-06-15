import math
from scipy import optimize


P_THRESHOLD = 0.02

TP_MIN = 0.1
TP_MAX = 100


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

    for D_MT in movements:
        D = D_MT[0]
        MT = D_MT[1]
        hit_prob = calc_hit_prob(D, W, MT, TP)
        fc_prob *= hit_prob

    # print('' + str(TP) + ' | ' + str(fc_prob))
    return fc_prob


# calculates the probability a player with throughput TP can hit an object
# with diameter W at distance D within movement time MT
def calc_hit_prob(D, W, MT, TP):
    
    a = 0.00

    if D == 0:
        return 1.0

    if MT*TP > 100:
        return 1.0

    if MT-a <= 0:
        return 0.0

    return math.erf(2.066 * (W/D) * (2**((MT-a)*TP)-1) / 2**0.5)


def calc_IP(D, W, MT):
    return math.log2(D/W + 1) / MT
