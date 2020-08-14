const TfModel = require("./TensorflowjsModel");
const MathjsModel = require("./MathjsModel");

list = [
  [0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
  [0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
  [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
  [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
  [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
  [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
  [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
];

function msToTime(s) {
  const ms = s % 1000;
  s = (s - ms) / 1000;
  const secs = s % 60;
  s = (s - secs) / 60;
  const mins = s % 60;
  const hrs = (s - mins) / 60;

  return hrs + ":" + mins + ":" + secs + "." + ms;
}

function durationTest(model) {
  let now = new Date();
  model.train();
  let duration = new Date();
  duration -= now;
  duration = msToTime(duration);
  return {
    duration,
    prediction: model.predict(),
  };
}

tfModel = new TfModel(list);
mathjsModel = new MathjsModel(list);

result1 = durationTest(tfModel)
result2 = durationTest(mathjsModel)

console.log(`Tensorflow model duration: ${result1.duration}`) 
console.log(`Mathjs lmodel duration: ${result2.duration}`)

/*
Using mathjs => Duration: 0:3:2.916
Using tfjs   => Duration: 0:3:1.835
 */
