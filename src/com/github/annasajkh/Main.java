package com.github.annasajkh;

import java.util.Random;

public class Main
{

    public static void main(String[] args)
    {
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 15, 1, 1);
        neuralNetwork.setActivationFunctions(ActivationFunctions.leakyRelu,ActivationFunctions.sigmoid);
        neuralNetwork.setLearningRate(0.1f);
        Random random = new Random();

        for(int i = 0; i < 10_000; i++)
        {
            float[] inputs = {random.nextInt(2), random.nextInt(2)};
            float[] outputs = {(int) inputs[0] ^ (int) inputs[1]};
            neuralNetwork.train(inputs, outputs);
        }

        neuralNetwork.save("model.txt");

        NeuralNetwork neuralNetworkFromSave = NeuralNetwork.load("model.txt");
        neuralNetworkFromSave.setActivationFunctions(ActivationFunctions.leakyRelu,ActivationFunctions.sigmoid);

        System.out.println("real answer is " + (1 ^ 1));
        System.out.println("AI prediction is " + neuralNetworkFromSave.process(new float[]{1, 1})[0]);
        System.out.println("real answer is " + (1 ^ 0));
        System.out.println("AI prediction is " + neuralNetworkFromSave.process(new float[]{1, 0})[0]);
        System.out.println("real answer is " + (0 ^ 1));
        System.out.println("AI prediction is " + neuralNetworkFromSave.process(new float[]{0, 1})[0]);
        System.out.println("real answer is " + (0 ^ 0));
        System.out.println("AI prediction is " + neuralNetworkFromSave.process(new float[]{0, 0})[0]);

    }
}
