/** Learn a simple linear function */

import * as sm from "@shumai/shumai";

// TODO(aduffy): Replace with modules
class LinearModel {
  private _n: number;

  public weights: sm.Tensor;

  constructor(n: number) {
    this._n = n;
    this.weights = sm.randn([n]).requireGrad();
  }

  // Expects to be passed an n-element tensor of features for X and
  // a 1-element tensor for Y.
  train(X: sm.Tensor, Y: sm.Tensor) {
    const y_hat = this.weights.matmul(X);
    const loss = sm.loss.mse(y_hat, Y);
    const ts = loss.backward() as Record<
      string,
      { grad: sm.Tensor; tensor: sm.Tensor }
    >;
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

const result = model.predict(sm.tensor(new Float32Array([1, 2, 3])));
console.log("result", result.toFloat32Array());
console.log("learned params", model.weights.toFloat32Array());
