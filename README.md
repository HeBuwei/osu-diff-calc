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
