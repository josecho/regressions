require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"]
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3, // try value: 3 too
  batchSize: 10
  // stochastic Greadient Descent -> batchSize: 1
});

regression.train();

//const r2 = regression.test(testFeatures, testLabels);
plot({
  x: regression.mseHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Mean Squared Error"
});
// console.log("R2 is", r2);

regression.predict([[130, 1.752, 307]]).print();
