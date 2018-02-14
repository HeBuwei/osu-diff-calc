import math

P_THRESHOLD = 0.1
LOG_TP_PRECISION = 1e-7


# calculates the probability a player with throughput TP can hit an object
# with diameter W at distance D within movement time MT
def calculate_hit_prob(D, W, MT, TP):
    
    a = 0.0

    if D == 0:
        return 1.0

    if MT*TP > 100:
        return 1.0

    if MT-a <= 0:
        return 0.0

    return math.erf(2.066 * (W/D) * (2**((MT-a)*TP)-1) / 2**0.5)


def calculate_fc_prob(Ds_MTs, W, TP):
    fc_prob = 1.0

    for (D, MT) in Ds_MTs:
        hit_prob = calculate_hit_prob(D, W, MT, TP)
        fc_prob *= hit_prob

    # print('' + str(TP) + ' | ' + str(fc_prob))
    return fc_prob


def calculate_throughput(Ds_MTs, W):
    log_TP_min = -3.0
    log_TP_max = 5.0

    fc_prob_TP_min = calculate_fc_prob(Ds_MTs, W, math.exp(log_TP_min))
    
    # if map is so easy that players with minimum throughput can fc with decent possibility
    if fc_prob_TP_min > P_THRESHOLD:
        return math.exp(log_TP_min)

    fc_prob_TP_max = calculate_fc_prob(Ds_MTs, W, math.exp(log_TP_max))

    # if map is too hard 
    if fc_prob_TP_max < P_THRESHOLD:
        return math.exp(log_TP_max)

    # binary search for a throughput that corresponds to an fc probability of P_THRESHOLD
    while log_TP_max - log_TP_min > LOG_TP_PRECISION:
        
        log_TP_mid = (log_TP_min + log_TP_max) / 2
        
        prob = calculate_fc_prob(Ds_MTs, W, math.exp(log_TP_mid))

        if prob > P_THRESHOLD:
            log_TP_max = log_TP_mid
        else:
            log_TP_min = log_TP_mid

    return math.exp((log_TP_min + log_TP_max) / 2)


def calculate_IP(D, W, MT):
    return math.log2(D/W + 1) / MT
