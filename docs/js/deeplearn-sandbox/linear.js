// import {Array1D, NDArrayMathGPU, Scalar} from 'deeplearn';
const { Array1D, NDArrayMathGPU, Scalar } = deeplearn;
const { Graph, Array2D } = deeplearn;
const { CostReduction, InCPUMemoryShuffledInputProviderBuilder, Session, SGDOptimizer, AdamOptimizer } = deeplearn;

// y = b * x + c

const graph = new Graph();
const x = graph.placeholder('x', []);

const b = graph.variable('b', Scalar.new(Math.random()));
const c = graph.variable('c', Scalar.new(Math.random()));

const y = graph.add(graph.multiply(b, x), c);

const yLabel = graph.placeholder('y label', []);
const cost = graph.meanSquaredCost(y, yLabel);

async function collectValues(costValue) {

  const params = {
    b: await session.eval(b).get(),
    c: await session.eval(c).get(),
    loss: await await costValue.get()
  };
  return params;
}

async function fitCurveThroughPoints(points, checkpointCallback) {
  const session = new Session(graph, math);

  // For more information on scope / track, check out the [tutorial on performance](/docs/tutorials/performance.html).
  return await math.scope(async (keep, track) => {

    /**
     * Inference
     */
    // Now we ask the graph to evaluate (infer) and give us the result when
    // providing a value 4 for "x".
    // NOTE: "a", "b", and "c" are randomly initialized, so this will give us
    // something random.
    let result =
      session.eval(y, [{ tensor: x, data: track(Scalar.new(4)) }]);
    // console.log(result.shape);
    console.log('y value for x = 4 before training: ', await result.get());

    /**
     * Training
     */
    // Now let's learn the coefficients of this quadratic given some data.
    // To do this, we need to provide examples of x and y.
    // The values given here are for values a = 3, b = 2, c = 1, with random
    // noise added to the output so it's not a perfect fit.

    const xs = [];
    const ys = [];
    for (const [x,y] of points) {
      xs.push(track(Scalar.new(x)));
      ys.push(track(Scalar.new(y)));
    }

    // When training, it's important to shuffle your data!
    const shuffledInputProviderBuilder =
      new InCPUMemoryShuffledInputProviderBuilder([xs, ys]);
    const [xProvider, yProvider] =
      shuffledInputProviderBuilder.getInputProviders();

    // Training is broken up into batches.
    const NUM_BATCHES = 100;
    const BATCH_SIZE = xs.length;
    // Before we start training, we need to provide an optimizer. This is the
    // object that is responsible for updating weights. The learning rate param
    // is a value that represents how large of a step to make when updating
    // weights. If this is too big, you may overstep and oscillate. If it is too
    // small, the model may take a long time to train.
    // https://github.com/PAIR-code/deeplearnjs/releases/tag/v0.3.0
    // const LEARNING_RATE = .01;
    // const optimizer = new SGDOptimizer(LEARNING_RATE);
    // http://cs231n.github.io/neural-networks-3/#ada
    // https://distill.pub/2017/momentum/
    const LEARNING_RATE = 20;
     // typically you do not touch these
    const beta1 = .9; // (decay rate: typical values are [0.9, 0.99, 0.999])
    const beta2 = 0.999;
    const optimizer = new AdamOptimizer(LEARNING_RATE, beta1, beta2);

    const checkpointSteps = 1;

    let costValue;
    for (let i = 0; i < NUM_BATCHES; i++) {
      // Train takes a cost tensor to minimize; this call trains one batch and
      // returns the average cost of the batch as a Scalar.
      costValue = session.train(
        cost,
        // Map input providers to Tensors on the graph.
        [{ tensor: x, data: xProvider }, { tensor: yLabel, data: yProvider }],
        BATCH_SIZE, optimizer, CostReduction.MEAN);

      if (i % checkpointSteps === 0) {
        if (checkpointCallback) {
          const params = await collectValues(costValue);
          await checkpointCallback(params);
        } else {
          console.log('mean cost: ' + await costValue.get());
        }
      }
    }

    const params = await collectValues(costValue);
    console.log('params: ', params);
    return params;
  });
}