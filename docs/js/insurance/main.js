// import * as tf from '@tensorflow/tfjs';
(async () => {
  const parser = d3.dsvFormat(';');

  const r = await fetch(
    'https://raw.githubusercontent.com/DJCordhose/ai/master/notebooks/scipy/data/insurance-customers-1500.csv'
  );
  const csv = await r.text();
  const data = await parser.parse(csv);

  let inputData = data.map((entry) => {
    return [
      Number(entry.age),
      Number(entry['max speed']),
      Number(entry['thousand km per year'])
    ];
  });

  let groups = data.map((entry) => {
    return Number(entry.group);
  });

  const sampleInput = tf.tensor2d(inputData, [inputData.length, 3]);
  const sampleOutput = tf.oneHot(tf.tensor1d(groups, 'int32'), 3);

  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [3] }));
  model.add(tf.layers.dense({ units: 100, activation: 'tanh' }));
  model.add(tf.layers.dense({ units: 100, activation: 'tanh' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam',
    metrics: ['accuracy']
  });
  
  // Generate some synthetic data for training.
  const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7, 9], [5, 1]);

  // Train the model using the data.
  await model.fit(sampleInput, sampleOutput, {
    epochs: 140,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (...args) => {
        console.log(...args);
      },
      onEpochBegin: (iteration) => {
        // console.clear();
        // console.log(iteration);
      }
    }
  });
  model.save('localstorage://insurance');
  model.predict(tf.tensor2d([47, 160, 15], [1, 3])).print();
})();
