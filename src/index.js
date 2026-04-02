import  * as tf from '@tensorflow/tfjs-node';

//Dados simples: y = 2x - 1 em matrizes 2D
//Tensor é um tipo de array matemático do TensowFLow
//XS é os dados iniciais e YS as saídas esperadas que forma y=2x-1
const xs = tf.tensor([1, 2, 3, 4]);
const ys = tf.tensor([1, 3, 5, 7]);

async function train(epochs) {
  //Modelo sequêncial, rede neural camada por camada
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError'
  });


  //Roda treinos com os dados XS e YS por 200 épocas
  await model.fit(xs, ys, { epochs });
  const output = model.predict(tf.tensor([5]));
  return output.dataSync()[0]; 
}

async function main()
{
  var results = [];
  const epochList = [5, 50, 100, 200, 500];

  for(const e of epochList){
    const result = await train(e);
    results.push(e + ' => ' + result);
  }

  console.log(results);
}

main();
