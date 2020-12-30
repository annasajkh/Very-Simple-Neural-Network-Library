package com.github.annasajkh;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.io.*;

public class NeuralNetwork
{
    private DoubleMatrix[] network;
    private DoubleMatrix[] weights;
    private DoubleMatrix[] biases;
    private int inputSize;
    private int hiddenLayerSize;
    private int layerCount;
    private int outputSize;
    private double[] expectedOutput;
    private double learningRate = 0.01;

    private static DoubleMatrix sigmoid(DoubleMatrix x)
    {
        return MatrixFunctions.exp(x.neg())
                              .add(1)
                              .rdiv(1);
    }

    private static DoubleMatrix dsigmoid(DoubleMatrix y)
    {
        //assuming y is has already been sigmoid
        return y.mul(y.rsub(1));
    }

    public NeuralNetwork(int inputSize, int hiddenLayerSize, int outputSize)
    {
        this(inputSize, hiddenLayerSize, outputSize, 1);
    }

    public NeuralNetwork(NeuralNetwork neuralNetwork)
    {
        this(neuralNetwork.inputSize, neuralNetwork.hiddenLayerSize, neuralNetwork.outputSize, neuralNetwork.layerCount);
        setLearningRate(neuralNetwork.learningRate);
        weights = neuralNetwork.weights;
        biases = neuralNetwork.biases;
    }

    public NeuralNetwork(int inputSize, int hiddenLayerSize, int outputSize, int layerCount)
    {
        //make new array of matrix
        weights = new DoubleMatrix[1 + layerCount];
        biases = new DoubleMatrix[1 + layerCount];
        this.inputSize = inputSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.layerCount = layerCount;
        this.outputSize = outputSize;

        //make layers of hidden layer as many as the layer count
        DoubleMatrix hiddenLayers = new DoubleMatrix(hiddenLayerSize);

        //make the network with size of input + layerCount + output
        network = new DoubleMatrix[layerCount + 2];

        //make network index 0 the size of input cuz it's a input layer
        network[0] = new DoubleMatrix(inputSize);

        //fill network index 1 - ? with the hidden layer
        for (int i = 1; i < network.length - 1; i++)
        {
            network[i] = hiddenLayers;
        }

        //make network index last the size of output cuz it's a output layer
        network[network.length - 1] = new DoubleMatrix(outputSize);

        for (int i = 1; i < network.length; i++)
        {

            DoubleMatrix weight;
            DoubleMatrix bias;

            //make this if it's a hidden layer
            if (i > 1 && i != network.length - 1)
            {
                //weight from hidden to hidden
                weight = DoubleMatrix.rand(hiddenLayerSize, hiddenLayerSize)
                                     .mul(2)
                                     .sub(1);
            }
            else
            {
                //make this is it's a output layer
                if (i == network.length - 1)
                {
                    //weight from hidden to output
                    weight = DoubleMatrix.rand(outputSize, hiddenLayerSize)
                                         .mul(2)
                                         .sub(1);
                }
                //make this is it's a input layer
                else
                {
                    //weight from input to hidden
                    weight = DoubleMatrix.rand(hiddenLayerSize, inputSize)
                                         .mul(2)
                                         .sub(1);
                }
            }

            //set weight array to matrix
            weights[i - 1] = weight;

            //if is not output layer make new matrix size of hidden layer else make new matrix size of output layer
            if (i != network.length - 1)
            {
                bias = DoubleMatrix.rand(hiddenLayerSize, 1)
                                   .mul(2)
                                   .sub(1);
            }
            else
            {
                bias = DoubleMatrix.rand(outputSize, 1)
                                   .mul(2)
                                   .sub(1);
            }

            biases[i - 1] = bias;
        }
    }

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    private void preprocess(int i, boolean train)
    {

        //matrix multiplacation between weight and before layer and also adding the biases and pass it to sigmoid function
        network[i] = sigmoid(weights[i - 1].mmul(network[i - 1])
                                           .add(biases[i - 1]));

        //if the current layer is the output and it's training then do the backpropagation
        if (i == network.length - 1 && train)
        {
            backpropagation(network[i]);
        }
    }

    public double[] process(double[] input)
    {
        //pass input to input layer
        network[0] = new DoubleMatrix(input);

        //feed forward the input from layer 1 so it can get layer 0
        for (int i = 1; i < network.length; i++)
        {
            preprocess(i, false);
        }

        //return the last layer aka the output
        return network[network.length - 1].toArray();
    }

    private DoubleMatrix[] getAllErrors(DoubleMatrix error)
    {
        //make array of matrix and set last index = error
        DoubleMatrix[] errors = new DoubleMatrix[weights.length];
        errors[errors.length - 1] = error;

        //calculate error on index i and pass it on index before it
        for (int i = errors.length - 1; i >= 1; i--)
        {
            errors[i - 1] = weights[i].transpose()
                                      .mmul(errors[i]);
        }

        return errors;
    }


    private void changingWeightsAndBiases(int index, DoubleMatrix errors, DoubleMatrix layer, DoubleMatrix afterLayer)
    {

        //calculate gradient (layer * (1 - layer)) and multiply it by errors and learning rate
        DoubleMatrix gradient = dsigmoid(layer).mul(errors)
                                               .mul(learningRate);

        //deltaWeight = gradient multiply afterLayer transposted
        DoubleMatrix deltaWeight = gradient.mmul(afterLayer.transpose());

        //adjust the weight by deltaWeight
        weights[index] = weights[index].add(deltaWeight);

        //adjust the bias by it's delta (it's just the gradient)
        biases[index] = biases[index].add(gradient);

    }

    private void backpropagation(DoubleMatrix output)
    {

        //calculate output error
        DoubleMatrix error = new DoubleMatrix(expectedOutput).sub(output);

        //get all errors
        DoubleMatrix[] errors = getAllErrors(error);

        //backpropagation
        for (int i = errors.length - 1; i >= 0; i--)
        {
            changingWeightsAndBiases(i, errors[i], network[i + 1], network[i]);
        }
    }

    public void train(double[] input, double[] expectedOutput)
    {
        //checking if expected output length is greater than output size
        if (expectedOutput.length > outputSize)
        {
            System.out.println("Error expected output is bigger than the output size");
            return;
        }

        //checking if input length is greater than input size
        if (expectedOutput.length > inputSize)
        {
            System.out.println("Error input is bigger than the input size");
            return;
        }

        this.expectedOutput = expectedOutput;

        //pass input to input layer
        network[0] = new DoubleMatrix(input);

        //feedforward
        for (int i = 1; i < network.length; i++)
        {
            preprocess(i, true);
        }

    }

    private DoubleMatrix matrixMutate(DoubleMatrix matrix)
    {
        return matrix.add(DoubleMatrix.rand(matrix.rows, matrix.columns)
                                      .mul(2)
                                      .sub(1));
    }

    //mutate weights by chance between 0 - 1
    public NeuralNetwork mutateWeights(double chance)
    {
        NeuralNetwork neuralNetwork = clone();
        if (Math.random() > chance)
        {
            return neuralNetwork;
        }
        for (int i = 0; i < neuralNetwork.weights.length; i++)
        {
            neuralNetwork.weights[i] = matrixMutate(neuralNetwork.weights[i]);
        }
        return neuralNetwork;
    }

    //mutate biases by chance between 0 - 1
    public NeuralNetwork mutateBiases(double chance)
    {
        NeuralNetwork neuralNetwork = clone();
        if (Math.random() > chance)
        {
            return neuralNetwork;
        }
        for (int i = 0; i < neuralNetwork.biases.length; i++)
        {
            neuralNetwork.biases[i] = matrixMutate(neuralNetwork.biases[i]);
        }
        return neuralNetwork;
    }

    //mutates weights and biases by chance between 0 - 1
    public NeuralNetwork mutate(double chance)
    {
        NeuralNetwork neuralNetwork = clone();
        if (Math.random() > chance)
        {
            return neuralNetwork;
        }
        for (int i = 0; i < neuralNetwork.weights.length; i++)
        {
            neuralNetwork.weights[i] = matrixMutate(neuralNetwork.weights[i]);
        }
        for (int i = 0; i < neuralNetwork.biases.length; i++)
        {
            neuralNetwork.biases[i] = matrixMutate(neuralNetwork.biases[i]);
        }
        return neuralNetwork;
    }

    public static NeuralNetwork load()
    {
        NeuralNetwork neuralNetwork = null;
        try
        {
            BufferedReader reader = new BufferedReader(new FileReader("data/config.txt"));
            String line = reader.readLine();
            reader.close();
            String[] args = line.split(",");
            neuralNetwork = new NeuralNetwork(Integer.parseInt(args[0]), Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
            neuralNetwork.setLearningRate(Double.parseDouble(args[4]));
            for (int i = 0; i < neuralNetwork.weights.length; i++)
            {
                neuralNetwork.weights[i].load("data/weight_" + i);
            }
            for (int i = 0; i < neuralNetwork.biases.length; i++)
            {
                neuralNetwork.biases[i].load("data/bias_" + i);
            }
        }
        catch (IOException exception)
        {
            exception.printStackTrace();
            return null;
        }
        return neuralNetwork;
    }

    public void saveFiles()
    {
        BufferedWriter writer;
        try
        {
            new File("data").mkdirs();
            writer = new BufferedWriter(new FileWriter("data/config.txt"));
            writer.append(inputSize + ",");
            writer.append(hiddenLayerSize + ",");
            writer.append(outputSize + ",");
            writer.append(layerCount + ",");
            writer.append(learningRate + "\n");
            writer.close();
            for (int i = 0; i < weights.length; i++)
            {
                weights[i].save("data/weight_" + i);
            }
            for (int i = 0; i < weights.length; i++)
            {
                biases[i].save("data/bias_" + i);
            }
        }
        catch (IOException exception)
        {
            exception.printStackTrace();
        }
    }

    public NeuralNetwork clone()
    {
        return new NeuralNetwork(this);
    }

}
