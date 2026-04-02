import  * as tf from '@tensorflow/tfjs-node';

//Dados simples: y = 2x - 1
const xs = tf.tensor([1, 2, 3, 4]);
const ys = tf.tensor([1, 3, 5, 7]);

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
});

async function train() {
  await model.fit(xs, ys, { epochs: 200 });

  const output = model.predict(tf.tensor([5]));
  return output.data(); //Deve ser perto de 9
}

async function main()
{
  const data = await train();
  console.log(data);
  const result = data[0];
  console.log(result);
}

main();
