# osu-diff-calc

#### Warning: This program is only a prototype and is not user friendly. 

## Installation
You will need Node.js and Python 3 to run this program. You will also need NumPy 1.14+, SciPy 1.3+ and MatPlotLib 2.2+

After cloning this repo, run
`npm install osu-parser` and `python install.py`

## Usage

#### Adding a beatmap
Copy the .osu file into `data/maps` folder and rename it to `<filename>.osu` so that the filename is easy to remember. After that, run `node parse.js <filename>`

#### Calculating the sr of one beatmap
`python diff_calc.py <filename>` to calculate the star rating of the beatmap. Only maps you have added can be calculated.
To calculate the sr with mods, run `python diff_calc.py <filename> <modstring>`, where `<modstring>` can be `hr`, `dt`, `ez`, `ht` and the concatenation of them.

#### Analyze beatmap using graphs
`python analyze.py <filename> <modstring>`


#### Calculating the sr of all beatmaps (without mods)
`python compare.py all`

#### Calculating the sr of a group of beatmap
First you need to update your `map_pack.json` so that it looks like this:
```
{  
  "<groupname>": [  
    ["<filename>", "<modstring>"],  
    ["<filename>", "<modstring>"],  
    ...  
  ],  
  "<groupname>": [  
    ["<filename>", "<modstring>"],  
    ["<filename>", "<modstring>"],  
    ...  
  ],  
  ...  
}
```
Then use `python compare.py <groupname>` to calculate the sr.

## Explanation of Algorithm

Instead of aim and speed, this algorithm divides player skill into aiming and tapping.

The aiming component covers all types of aiming action, be it snap aim or flow aim. This component
calculates aim difficulty by finding the player skill at which the probability of FC is equal to a constant (0.02 for now). This component makes use of Fitts' Law - an empirical formula that relates target width, distance to target, movement time, human's aiming skill and success probability.

Similar to the current speed skill, the tapping skill finds the strain at each circle. The strain value decays exponentially with time. However the speed skill and the tapping skill differ in the following 3 points:

1. The speed skill considers the distance (in position) between circles, while the tapping skill does not.

2. The speed skill uses a single strain value, while the tapping skill uses a few strain values with different decay rate.

3. The speed skill is aggregated by calculating the weighted sum of strain value over all sections, while the tapping skill is aggregated by simply taking the maximum.
