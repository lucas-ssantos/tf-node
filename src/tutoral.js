import  * as tf from '@tensorflow/tfjs-node';

//Dados simples: y = 2x - 1 em matrizes 2D
//Tensor é um tipo de array matemático do TensowFLow
//XS é os dados iniciais e YS as saídas esperadas que forma y=2x-1
const xs = tf.tensor([1, 2, 3, 4]);
const ys = tf.tensor([1, 3, 5, 7]);

//Modelo sequêncial, rede neural camada por camada
const model = tf.sequential();

//Dense é camada totalmente conectada
//Units é 1 neurônio
//inputShape é a entrada de apenas um número
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

//Optimizer é o método de ajuste dos pesos
//Loss mede o erro de previsão
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
});

async function train() {
  //Roda treinos com os dados XS e YS por 200 épocas
  await model.fit(xs, ys, { epochs: 200 });

  //Pega o modelo e treina roda com o valor 5
  const output = model.predict(tf.tensor([5]));
  return output.data(); 
  //Output é um objeto Tensor
  //O dado resultado que queremos está onde Judas perdeu as botas...
  //então precisa de algumas funções do objeto para pegar o resultado em si

  //EXEMPLO DE SAÍDAS com um tensor 2D com valor 9
  /*
   const t = tf.tensor([[9]]); //Matriz 2D 1x1 
   conslo.log(t); 		//Tensor { shape: [1,1], dtype: 'float32' }
   t.print(); 			//[[9]] saí a matrix com o resultado
   console.log(t.dataSync()); 	//Float32Array [9] *
   console.log(t.data[0])	//9

   				*Bom para pegar em logs
  */

  //Deve ser perto de 9, pois 2 * 5 - 1 = 9
  //Não vai dar exatos 9, pq IA é a IA... e.e
}

async function main()
{
  const data = await train();
  const result = data[0];
  console.log(result);
}

main();
