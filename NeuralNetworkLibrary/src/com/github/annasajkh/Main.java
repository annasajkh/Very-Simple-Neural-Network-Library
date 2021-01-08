package com.github.annasajkh;

import java.util.Arrays;
import java.util.Random;

public class Main
{

    public static void main(String[] args)
    {


        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 5, 1,2);
        neuralNetwork.setLearningRate(0.1);
        Random random = new Random();
        for (int i = 0; i < 100_000; i++)
        {
            double[] inputs = {random.nextInt(2), random.nextInt(2)};
            double[] outputs = {(int) inputs[0] ^ (int) inputs[1]};
            neuralNetwork.train(inputs, outputs);
        }

        System.out.println(1 ^ 1);
        System.out.println(Arrays.toString(neuralNetwork.process(new double[]{1, 1})));
        System.out.println(1 ^ 0);
        System.out.println(Arrays.toString(neuralNetwork.process(new double[]{1, 0})));
        System.out.println(0 ^ 1);
        System.out.println(Arrays.toString(neuralNetwork.process(new double[]{0, 1})));
        System.out.println(0 ^ 0);
        System.out.println(Arrays.toString(neuralNetwork.process(new double[]{0, 0})));
        

    }
}
