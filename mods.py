import sys

def str_to_mods(s):

    mod_abbrs = [s[i:i+2] for i in range(0, len(s), 2)]

    if 'hr' in mod_abbrs:
        hr_ez = 'hr'
    elif 'ez' in mod_abbrs:
        hr_ez = 'ez'
    else:
        hr_ez = 'nm'

    if 'dt' in mod_abbrs:
        dt_ht = 'dt'
    elif 'ht' in mod_abbrs:
        dt_ht = 'ht'
    else:
        dt_ht = 'nm'

    return [hr_ez, dt_ht]


def mods_to_str(mods):

    s = ""

    if mods[0] == 'hr' or mods[0] == 'ez':
        s += mods[0]

    if mods[1] == 'dt' or mods[1] == 'ht':
        s += mods[1]

    return s


if __name__ == "__main__":
    
    s = sys.argv[1]
    print(str_to_mods(s))
