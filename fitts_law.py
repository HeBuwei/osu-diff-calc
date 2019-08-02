import math


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
