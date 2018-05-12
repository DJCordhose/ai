/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';

/**
 * We want to learn the coefficients that give correct solutions to the
 * following quadratic equation:
 *      y = a * x^3 + b * x^2 + c * x + d
 * In other words we want to learn values for:
 *      a
 *      b
 *      c
 *      d
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */

// Step 1. Set up variables, these are the things we want the model
// to learn in order to do prediction accurately. We will initialize
// them with random values.
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model perfoms.
// https://js.tensorflow.org/api/0.6.1/#Training-Optimizers
let optimizer;
{
  // http://cs231n.github.io/neural-networks-3/#sgd
  // https://js.tensorflow.org/api/0.6.1/#train.sgd
  const learningRate = 0.5;
  const sgdOptimizer = tf.train.sgd(learningRate);
  optimizer = sgdOptimizer;
}
{
  // https://distill.pub/2017/momentum/
  // https://js.tensorflow.org/api/0.6.1/#train.momentum
  const LEARNING_RATE = 0.1;
  const MOMENTUM = 0.9; // possibly good values [0.5, 0.9, 0.95, 0.99]
  momentumOptimizer = new tf.train.momentum(LEARNING_RATE, MOMENTUM);
  // optimizer = momentumOptimizer;
}
{
  // http://cs231n.github.io/neural-networks-3/#ada
  // https://js.tensorflow.org/api/0.6.1/#train.adam
  const LEARNING_RATE = 0.1;
  const adamOptimizer = new tf.train.adam(LEARNING_RATE);
  // optimizer = adamOptimizer;
}

// Step 3. Write our training process functions.

/*
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 *
 * @return number predicted y value
 */
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a
      .mul(x.pow(tf.scalar(3, "int32")))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

/*
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 */
function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  const error = prediction
    .sub(labels)
    .square()
    .mean();
  return error;
}

/*
 * This will iteratively train our model.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(trainingData, numIterations, checkpointCallback) {
  for (let iter = 0; iter < numIterations; iter++) {
    // optimizer.minimize is where the training happens.

    // The function it takes must return a numerical estimate (i.e. loss)
    // of how well we are doing using the current state of
    // the variables we created at the start.

    // This optimizer does the 'backward' step of our training process
    // updating variables defined previously in order to minimize the
    // loss.
    optimizer.minimize(() => {
      // Feed the examples into the model
      const pred = predict(trainingData.xs);
      return loss(pred, trainingData.ys);
    });

    await checkpointCallback({ a, b, c, d }, trainingData);

    // Use tf.nextFrame to not block the browser.
    await tf.nextFrame();
  }
}

async function learnCoefficients() {
  const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
  const trainingData = generateData(100, trueCoefficients);
  const numIterations = 200;

  // Plot original data
  renderCoefficients("#data .coeff", trueCoefficients);
  await plotData("#data .plot", trainingData.xs, trainingData.ys);

  // See what the predictions look like with random coefficients
  renderCoefficients("#random .coeff", {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0]
  });
  const predictionsBefore = predict(trainingData.xs);
  await plotDataAndPredictions(
    "#random .plot",
    trainingData.xs,
    trainingData.ys,
    predictionsBefore
  );
  predictionsBefore.dispose();

  // Train the model!
  await train(trainingData, numIterations, predictAndPlot);
  await predictAndPlot({ a, b, c, d }, trainingData);
}

async function predictAndPlot({ a, b, c, d }, trainingData) {
  renderCoefficients("#trained .coeff", {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0]
  });
  const predictionsAfter = predict(trainingData.xs);
  await plotDataAndPredictions(
    "#trained .plot",
    trainingData.xs,
    trainingData.ys,
    predictionsAfter
  );
  predictionsAfter.dispose();
}

learnCoefficients();
