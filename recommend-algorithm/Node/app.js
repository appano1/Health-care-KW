const {PythonShell} = require('python-shell');

list = [[0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
        [0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
        [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0]];
userNumber = 3;

const options = {
    mode: 'text',
    pythonPath: '/opt/anaconda3/envs/tf1/bin/python3',
    pythonOptions: ['-u'],
    scriptPath: '/Users/leeseungcheol/GitHub/Health-care-KW/recommend-algorithm/',
    args: [list, userNumber]
};

PythonShell.run('recommend_test01.py',
    options, function(err, result) {
        if(err) throw err;

        // result.forEach(v => console.log(v));
        let result_recommendation = [];
        result.forEach(v => result_recommendation.push(Number(v)));
        console.log(result_recommendation);
    });
