/**
 * learn a simple linear function. Get familiar with the basic tensor operators
 * provided by @shumai.
 */
import * as sm from "@shumai/shumai";

class LinearModel {
  private _n: number;

  public weights: sm.Tensor;

  constructor(n: number) {
    this._n = n;
    this.weights = sm.randn([n]).requireGrad();
  }

  // Expects to be passed an M x N dimensional tensor of xs inputs,
  // and an N dimensional tensor of the ys ground-truth outputs.
  train(X: sm.Tensor, Y: sm.Tensor) {
    const y_hat = this.weights.matmul(X);
    const loss = sm.loss.mse(y_hat, Y);
    const ts = loss.backward() as Record<
      string,
      { grad: sm.Tensor; tensor: sm.Tensor }
    >;

    // Run the optimizer against the list of differentiable tensors.
    sm.optim.sgd(ts, 1e-3);
  }

  predict(X: sm.Tensor): sm.Tensor {
    return this.weights.detach().matmul(X);
  }
}

const model = new LinearModel(3);

// See if we can learn the function f([a, b, c]) = 2a + 3b - c
const hidden_weights = sm.tensor(new Float32Array([2, 3, -1]));

console.log(hidden_weights.shape);

console.log("begin training...");
for (let i = 0; i < 10000; i++) {
  const X = sm.randn([3]);
  const Y = X.matmul(hidden_weights);
  // console.log(Y.shape);
  model.train(X, Y);
}
console.log("training complete");

// See if we can apply forward and get the expected result

const result = model.predict(sm.tensor(new Float32Array([1, 2, 3])));
console.log(result.toFloat32Array());
console.log(model.weights.toFloat32Array());

// Have our dumb little holdout set here as well.
