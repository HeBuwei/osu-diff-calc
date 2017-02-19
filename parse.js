parser = require('osu-parser');
fs = require('fs');

name = process.argv[2];

parser.parseFile('data/maps/' + name + '.osu', function (err, beatmap) {
	fs.writeFile('data/maps/' + name + '.json', JSON.stringify(beatmap, null, 2));
});
