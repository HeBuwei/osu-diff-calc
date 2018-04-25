import math
from scipy import optimize


P_THRESHOLD = 0.1
LOG_TP_PRECISION = 1e-7


# calculates the probability a player with throughput TP can hit an object
# with diameter W at distance D within movement time MT
def calc_hit_prob(D, W, MT, TP):
    
    a = 0.0

    if D == 0:
        return 1.0

    if MT*TP > 100:
        return 1.0

    if MT-a <= 0:
        return 0.0

    return math.erf(2.066 * (W/D) * (2**((MT-a)*TP)-1) / 2**0.5)


def calc_fc_prob(TP, Ds_MTs, W):
    fc_prob = 1.0

    for (D, MT) in Ds_MTs:
        hit_prob = calc_hit_prob(D, W, MT, TP)
        fc_prob *= hit_prob

    # print('' + str(TP) + ' | ' + str(fc_prob))
    return fc_prob


def calc_throughput(Ds_MTs, W):
    log_TP_min = -3.0
    log_TP_max = 5.0

    fc_prob_TP_min = calc_fc_prob(math.exp(log_TP_min), Ds_MTs, W)
    
    # if map is so easy that players with minimum throughput can fc with decent possibility
    if fc_prob_TP_min >= P_THRESHOLD:
        return math.exp(log_TP_min)

    fc_prob_TP_max = calc_fc_prob(math.exp(log_TP_max), Ds_MTs, W)

    # if map is too hard 
    if fc_prob_TP_max <= P_THRESHOLD:
        return math.exp(log_TP_max)

    # # binary search for a throughput that corresponds to an fc probability of P_THRESHOLD
    # while log_TP_max - log_TP_min > LOG_TP_PRECISION:
        
    #     log_TP_mid = (log_TP_min + log_TP_max) / 2
        
    #     prob = calc_fc_prob(Ds_MTs, W, math.exp(log_TP_mid))

    #     if prob > P_THRESHOLD:
    #         log_TP_max = log_TP_mid
    #     else:
    #         log_TP_min = log_TP_mid

    #     count = count + 1

    def fc_prob_minus_threshold(TP, Ds_MTs, W):
        return calc_fc_prob(TP, Ds_MTs, W) - P_THRESHOLD

    x, r = optimize.brentq(fc_prob_minus_threshold, math.exp(log_TP_min), math.exp(log_TP_max), args=(Ds_MTs, W), full_output=True)

    print(r)

    # return math.exp((log_TP_min + log_TP_max) / 2)
    return x


def calc_IP(D, W, MT):
    return math.log2(D/W + 1) / MT
