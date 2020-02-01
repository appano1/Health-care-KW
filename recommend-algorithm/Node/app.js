const {PythonShell} = require('python-shell');

const options = {
    mode: 'text',
    pythonPath: '/opt/anaconda3/envs/tf1/bin/python3',
    pythonOptions: ['-u'],
    scriptPath: '',
    args: ['value1', 'value2', 'value3']
};

PythonShell.run('/Users/leeseungcheol/GitHub/Health-care-KW/recommend-algorithm/recommend_test.py',
    options, function(err, result) {
        if(err) throw err;
        result.forEach(v => console.log(v));
    });
