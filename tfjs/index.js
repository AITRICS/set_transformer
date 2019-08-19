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

/**
 * Addition RNN example.
 *
 * Based on Python Keras example:
 *   https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

function createAndCompileModel(type, inputLength, hiddenSize, learningRate) {
  // Encoder
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: hiddenSize, activation: 'relu', name: 'EncoderFC1', inputShape: [inputLength, 1] }),
      tf.layers.dense({ units: hiddenSize, activation: 'relu', name: 'EncoderFC2' }),
      tf.layers.dense({ units: hiddenSize, activation: 'relu', name: 'EncoderFC3' }),
      tf.layers.dense({ units: hiddenSize, name: 'EncoderFC4' })
    ]
  });

  switch (type) {
    case 'MaxDeepset':
      model.add(tf.layers.maxPooling1d({ poolSize: inputLength, strides: 1, name: 'MaxPooling' }));
      model.add(tf.layers.flatten({ shape: [null, hiddenSize], name: 'Flatten' }));
      break;
    case 'MeanDeepset':
      model.add(tf.layers.averagePooling1d({ poolsize: inputLength, name: 'AveragePooling' }));
      model.add(tf.layers.flatten({ shape: [null, hiddenSize], name: 'Flatten' }));
      break;
    default:
      throw new Error('Model type is not specified!');
  }

  // Decoder
  model.add(tf.layers.dense({ units: hiddenSize, activataion: 'relu', name: 'DecoderFC1' }));
  model.add(tf.layers.dense({ units: 1, name: 'DecoderFC2' }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: tf.losses.absoluteDifference,
    metrics: ['accuracy']
  });
  return model;
}

// Need to implement for full set transformer functionality, skip for this demo
function layerNorm(inputs, beginNormAxis) {
  const inputsShape = inputs.shape
  const inputsRank = Number(inputs.rankType)
  const dataType = inputs.dtype

  if (beginNormAxis < 0) {
    beginNormAxis = inputsRank + beginNormAxis
  }

  const paramsShape = inputsShape.slice(-1)

  const beta = tf.variable(tf.zeros(paramsShape), true, null, dataType)
  const gamma = tf.variable(tf.ones(paramsShape), true, null, dataType)

  const normAxes = [];
  for (let i = beginNormAxis; i < inputsRank; ++i) {
    normAxes.push(i)
  }

  const { mean, variance } = tf.moments(inputs, normAxes, false)

  const epsilon = 1e-3
  // const outputs = tf.batchNorm(inputs, mean, variance, { offset: beta, scale: gamma, variance_epsilon: epsilon })
  console.log(mean.shape)
  console.log(beta.rank)
  const outputs = tf.batchNorm(inputs, mean, variance, beta, gamma, epsilon)
  // const outputs = tf.layers.batchNormalization({ epsilon: epsilon, center: beta, scale: gamma, trainable: true, dtype: dataType })

  return tf.reshape(outputs, inputsShape)
}

class MAB {
  constructor(dimQ, dimK, dimV, numHeads, ln) {
    this.dimQ = dimQ
    this.dimK = dimK
    this.dimV = dimV
    this.numHeads = numHeads
    this.ln = ln

    this.fc_q = tf.layers.dense({ units: dimV, name: 'fc_q' })
    this.fc_k = tf.layers.dense({ units: dimV, name: 'fc_k' })
    this.fc_v = tf.layers.dense({ units: dimV, name: 'fc_v' })

    this.fc_o = tf.layers.dense({ units: dimV, activation: 'relu', name: 'fc_o' })
  }

  forward(inpQ, inpK) {
    const batchSize = inpQ.shape[0]

    const Q = this.fc_q.apply(inpQ)
    const K = this.fc_k.apply(inpK)
    const V = this.fc_v.apply(inpK)

    console.log(Q)
    let x = tf.reshape(Q, (batchSize, -1, this.numHeads, Math.floor(Q.shape[2] / this.numHeads)))
    x = tf.transpose(x, [0, 2, 1, 3])
    console.log(x)

    const Q_ = tf.concat(tf.split(Q, this.numHeads, 2), 0)
    const K_ = tf.concat(tf.split(K, this.numHeads, 2), 0)
    const V_ = tf.concat(tf.split(V, this.numHeads, 2), 0)

    const A = tf.softmax(tf.div(tf.matMul(Q_, tf.transpose(K_, [0, 2, 1])), Math.sqrt(this.dimV)), 2)
    const O = tf.concat(tf.split(tf.add(Q_, tf.matMul(A, V_)), this.numHeads, 0), 2)
    const output = tf.add(O, this.fc_o.apply(O))
    return output;
  }
}

class SAB {
  constructor(dimIn, dimOut, numHeads, ln) {
    this.dimIn = dimIn
    this.dimOut = dimOut
    this.numHeads = numHeads
    this.ln = ln
    this.mab = new MAB(dimIn, dimIn, dimOut, numHeads, ln)
  }

  forward(inp) {
    return this.mab.forward(inp, inp)
  }
}

class PMA {
  constructor(dim, numHeads, numSeeds, ln) {
    const initializer = tf.initializers.glorotUniform(this.numSeeds)
    this.S = initializer.apply([1, numSeeds, dim])
    this.mab = new MAB(dim, dim, dim, numHeads, ln)
  }

  forward(inp) {
    return this.mab.forward(this.S.tile([inp.shape[0], 1, 1]), inp)
  }
}

function createAndCompileTransformer(inputLength, numHeads, learningRate) {
  // Encoder
  const x = tf.input({ shape: [inputLength, 1], sparse: true });

  const sab1 = new SAB(1, 64, numHeads, false)
  const sab2 = new SAB(64, 64, numHeads, false)
  const encodedO = sab2.forward(sab1.forward(x))

  // Decoder
  const pma = new PMA(64, numHeads, 1, false)
  const decodedO = pma.forward(encodedO)
  const O = tf.layers.dense({ units: 1, name: 'dense' }).apply(decodedO)

  const model = tf.model({ inputs: x, outputs: O });
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: tf.losses.absoluteDifference,
    metrics: ['accuracy']
  });
  return model;
}

function convertDataToTensors(data) {
  const xs = data.map(datum => datum[0]);
  const ys = data.map(datum => datum[1]);
  return [tf.expandDims(tf.tensor(xs), 2), tf.expandDims(tf.tensor(ys), 1)]
}

class MaxRegressionDemo {
  constructor(data, inputLength, trainigSize, hiddenSize, numHeads, learningRate) {
    const split = Math.floor(trainigSize * 0.9);
    this.trainData = data.slice(0, split);
    this.testData = data.slice(split);

    [this.trainXs, this.trainYs] = convertDataToTensors(this.trainData);
    [this.testXs, this.testYs] = convertDataToTensors(this.testData);

    this.maxDeepset = createAndCompileModel("MaxDeepset", inputLength, hiddenSize, learningRate);
    this.meanDeepset = createAndCompileModel("MeanDeepset", inputLength, hiddenSize, learningRate);
    // this.transformer = createAndCompileTransformer(inputLength, numHeads, learningRate);

    const maxModelSummaryContainer = document.getElementById('maxModelSummary');
    const meanModelSummaryContainer = document.getElementById('meanModelSummary');
    // const transformerContainer = document.getElementById('transformerSummary');

    tfvis.show.modelSummary(maxModelSummaryContainer, this.maxDeepset);
    tfvis.show.modelSummary(meanModelSummaryContainer, this.meanDeepset);
    // tfvis.show.modelSummary(transformerContainer, this.transformer);

    const valueDistContainer = document.getElementById('valueDist');
    tfvis.show.valuesDistribution(valueDistContainer, this.trainXs)
  }

  async train(iterations, batchSize) {
    const lossValues = [[], [], [], []];
    const accuracyValues = [[], [], [], []];
    for (let i = 0; i < iterations; ++i) {
      const beginMs = performance.now();

      const maxDeepsetHistory = await this.maxDeepset.fit(this.trainXs, this.trainYs, {
        epochs: 1,
        batchSize,
        validationData: [this.testXs, this.testYs],
        yiedlEvery: 'epoch'
      });

      const meanDeepsethistory = await this.meanDeepset.fit(this.trainXs, this.trainYs, {
        epochs: 1,
        batchSize,
        validationData: [this.testXs, this.testYs],
        yiedlEvery: 'epoch'
      });

      const elapsedMs = performance.now() - beginMs;
      const modelFitTime = elapsedMs / 1000;

      const maxDeepsetTrainLoss = maxDeepsetHistory.history['loss'][0];
      const maxDeepsetTrainAccuracy = maxDeepsetHistory.history['acc'][0];
      const maxDeepsetValLoss = maxDeepsetHistory.history['val_loss'][0];
      const maxDeepsetValAccuracy = maxDeepsetHistory.history['val_acc'][0];

      const meanDeepsetTrainLoss = meanDeepsethistory.history['loss'][0];
      const meanDeepsetTrainAccuracy = meanDeepsethistory.history['acc'][0];
      const meanDeepsetValLoss = meanDeepsethistory.history['val_loss'][0];
      const meanDeepsetValAccuracy = meanDeepsethistory.history['val_acc'][0];

      lossValues[0].push({ 'x': i, 'y': maxDeepsetTrainLoss });
      lossValues[1].push({ 'x': i, 'y': maxDeepsetValLoss });
      lossValues[2].push({ 'x': i, 'y': meanDeepsetTrainLoss });
      lossValues[3].push({ 'x': i, 'y': meanDeepsetValLoss });

      accuracyValues[0].push({ 'x': i, 'y': maxDeepsetTrainAccuracy });
      accuracyValues[1].push({ 'x': i, 'y': maxDeepsetValAccuracy });
      accuracyValues[2].push({ 'x': i, 'y': meanDeepsetTrainAccuracy });
      accuracyValues[3].push({ 'x': i, 'y': meanDeepsetValAccuracy });

      document.getElementById('trainStatus').textContent =
        `Iteration ${i + 1} of ${iterations}: ` +
        `Time per iteration: ${modelFitTime.toFixed(3)} (seconds)`;

      const lossContainer = document.getElementById('lossChart');
      tfvis.render.linechart(
        lossContainer,
        {
          values: lossValues,
          series: ['Max Deepset Train', 'Max Deepset Validation', 'Mean Deepset Train', 'Mean Deepset Validation']
        },
        {
          width: 420,
          height: 300,
          xLabel: 'epoch',
          yLabel: 'loss',
        });

      const accuracyContainer = document.getElementById('accuracyChart');
      tfvis.render.linechart(
        accuracyContainer,
        {
          values: accuracyValues,
          series: ['Max Deepset Train', 'Max Deepset Validation', 'Mean Deepset Train', 'Mean Deepset Validation']
        },
        {
          width: 420,
          height: 300,
          xLabel: 'epoch',
          yLabel: 'accuracy',
        });
    }
  }
}

function generateSetData(batchSize, maxLength) {
  return tf.tidy(() => {
    const length = tf.randomUniform([1,], 1, maxLength + 1, 'int32');
    const lengthData = length.dataSync();

    const x = tf.randomUniform([batchSize, lengthData[0]], 1, 100, 'int32');
    const y = tf.max(x, 1);

    return { x, y, lengthData };
  })
}

async function RunMaxRegressionDemo() {
  // hyperparameters
  const trainIterations = +(document.getElementById('trainIterations')).value;
  const batchSize = +(document.getElementById('batchSize')).value;
  const learningRate = +(document.getElementById('learningRate')).value;
  const hiddenSize = +(document.getElementById('hiddenSize')).value;
  const numHeads = 4;

  // generate data
  const trainingData = generateSetData(batchSize, 10);

  const x = trainingData.x.dataSync();
  const labels = trainingData.y.dataSync();
  const inputLength = trainingData.lengthData[0];

  var data = [];
  var start = 0;
  for (let i = 0; i < batchSize; i++) {
    data.push(x.slice(start, start + inputLength))
    start = start + inputLength
  }

  const dataLabelPairs = data.map(function (e, i) {
    return [e, labels[i]]
  })

  document.getElementById('trainModel').addEventListener('click', async () => {
    const demo = new MaxRegressionDemo(dataLabelPairs, inputLength, batchSize, hiddenSize, numHeads, learningRate);
    await demo.train(trainIterations, batchSize);
  });
}


RunMaxRegressionDemo();
