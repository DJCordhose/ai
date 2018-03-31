// import * as tf from '/node_modules/@tensorflow/tfjs/dist/index.js';

// https://js.tensorflow.org/tutorials/core-concepts.html


(async () => {
  // Define constants: y = 2x^2 + 4x + 8
  const a = tf.scalar(2);
  const b = tf.scalar(4);
  const c = tf.scalar(8);

  function predict(input) {
    // y = a * x ^ 2 + b * x + c
    // More on tf.tidy in the next section
    return tf.tidy(() => {
      const x = tf.scalar(input);

      const ax2 = a.mul(x.square());
      const bx = b.mul(x);
      const y = ax2.add(bx).add(c);

      return y;
    });
  }

  // Predict output for input of 2
  const result = predict(2);
  result.print() // Output: 24

})();
// const shape = [2, 3]; // 2 rows, 3 columns
// const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
// a.print(); // print Tensor values

// // The shape can also be inferred:
// const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
// b.print();

// const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
// c.print();

// const zeros = tf.zeros([3, 5]);
// zeros.print();

// const initialValues = tf.zeros([5]);
// const biases = tf.variable(initialValues); // initialize biases
// biases.print(); // output: [0, 0, 0, 0, 0]

// const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
// biases.assign(updatedValues); // update values of biases
// biases.print(); // output: [0, 1, 0, 1, 0]

// const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
// const d_squared = d.square();
// d_squared.print();

// const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
// const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

// const e_plus_f = e.add(f);
// e_plus_f.print();

// const sq_sum = e.add(f).square();
// sq_sum.print();

// // All operations are also exposed as functions in the main namespace,
// // so you could also do the following:
// const sq_sum_main = tf.square(tf.add(e, f));
// sq_sum_main.print();

