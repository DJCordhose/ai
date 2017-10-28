// import {Array1D, NDArrayMathGPU, Scalar} from 'deeplearn';
const {Array1D, NDArrayMathGPU, Scalar} = deeplearn

const math = new NDArrayMathGPU()
const a = Array1D.new([1, 2, 3])
const b = Scalar.new(2)

math.enableDebugMode()

// cleans up GPU resources after use
math.scope(() => {
  const result = math.add(a, b)
  console.log(result.getValues())  // Float32Array([3, 4, 5])
})
