import sys
import matplotlib.pyplot as plt


import diff_calc

if __name__ == "__main__":
    
    name = sys.argv[1]
    IPs, times = diff_calc.analyze_file_aim_diff('data/maps/' + name + '.json')
    
    plt.plot(times, IPs, '.')
    plt.show()
