(async () => {
  const model = await tf.loadModel('localstorage://insurance');
  model.predict(tf.tensor2d([47, 160, 15], [1, 3])).print();
})();
