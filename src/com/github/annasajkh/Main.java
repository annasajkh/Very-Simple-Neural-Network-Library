package com.github.annasajkh;

import java.math.BigDecimal;
import java.util.Random;

public class Main
{

    public static void main(String[] args) throws Exception
    {
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 10, 1, 3);
        neuralNetwork.setActivationFunctions(ActivationFunctions.leakyRelu,ActivationFunctions.sigmoid);
        neuralNetwork.setLearningRate(0.01f);
        Random random = new Random();

        for(int i = 0; i < 3_000; i++)
        {
            float[] inputs = {random.nextInt(2), random.nextInt(2)};
            float[] outputs = {(int) inputs[0] ^ (int) inputs[1]};
            neuralNetwork.train(inputs, outputs);
        }

        neuralNetwork.save("model.txt");

        NeuralNetwork neuralNetworkFromSave = NeuralNetwork.load("model.txt");
        neuralNetworkFromSave.setActivationFunctions(ActivationFunctions.leakyRelu,ActivationFunctions.sigmoid);

        System.out.println("real answer is " + (1 ^ 1));
        System.out.println("AI prediction is " + new BigDecimal(neuralNetworkFromSave.process(new float[]{1, 1})[0]));
        System.out.println("real answer is " + (1 ^ 0));
        System.out.println("AI prediction is " + new BigDecimal(neuralNetworkFromSave.process(new float[]{1, 0})[0]));
        System.out.println("real answer is " + (0 ^ 1));
        System.out.println("AI prediction is " + new BigDecimal(neuralNetworkFromSave.process(new float[]{0, 1})[0]));
        System.out.println("real answer is " + (0 ^ 0));
        System.out.println("AI prediction is " + new BigDecimal(neuralNetworkFromSave.process(new float[]{0, 0})[0]));

    }
}
