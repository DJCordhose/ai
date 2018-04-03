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

// import embed from 'vega-embed';
const embed = vegaEmbed;

const statusElement = document.getElementById("status");
const messageElement = document.getElementById("message");
const imagesElement = document.getElementById("images");

function isTraining() {
  statusElement.innerText = "Training...";
}
function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

function showTestResults(batch, predictions, labels) {
  statusElement.innerText = "Testing...";

  const testExamples = batch.xs.shape[0];
  let totalCorrect = 0;
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement("div");
    div.className = "pred-container";

    const canvas = document.createElement("canvas");
    draw(image.flatten(), canvas);

    const pred = document.createElement("div");

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;

    pred.className = `pred ${correct ? "pred-correct" : "pred-incorrect"}`;
    pred.innerText = `pred: ${prediction}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }
}

const lossLabelElement = document.getElementById("loss-label");
const accuracyLabelElement = document.getElementById("accuracy-label");
function plotLosses(lossValues) {
  embed(
    "#lossCanvas",
    {
      $schema: "https://vega.github.io/schema/vega-lite/v2.json",
      data: { values: lossValues },
      mark: {
        type: "line",
        orient: "vertical"
      },
      width: 260,
      orient: "vertical",
      encoding: {
        x: { field: "batch", type: "quantitative" },
        y: { field: "loss", type: "quantitative" },
        color: { field: "set", type: "nominal", legend: null }
      }
    },
    { width: 360 }
  );
  lossLabelElement.innerText =
    "last loss: " + lossValues[lossValues.length - 1].loss.toFixed(2);
}

function plotAccuracies(accuracyValues) {
  embed(
    "#accuracyCanvas",
    {
      $schema: "https://vega.github.io/schema/vega-lite/v2.json",
      data: { values: accuracyValues },
      width: 260,
      mark: { type: "line", legend: null, orient: "vertical" },
      orient: "vertical",
      encoding: {
        x: { field: "batch", type: "quantitative" },
        y: { field: "accuracy", type: "quantitative" },
        color: { field: "set", type: "nominal", legend: null }
      }
    },
    { width: 360 }
  );
  accuracyLabelElement.innerText =
    "last accuracy: " +
    (accuracyValues[accuracyValues.length - 1].accuracy * 100).toFixed(2) +
    "%";
}

function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
