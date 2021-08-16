package com.github.annasajkh;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;


public class NeuralNetwork
{
    private Matrix[] network;
    private Matrix[] weights;
    private Matrix[] biases;
    private int inputSize;
    private int hiddenLayerSize;
    private int outputSize;
    private float[] expectedOutput;
    private float learningRate = 0.01f;
    private int hiddenLayerCount;
    private ActivationFunction hiddenActivation = ActivationFunctions.leakyRelu;
    private ActivationFunction outputActivation = ActivationFunctions.sigmoid;


    public NeuralNetwork(int inputSize, int hiddenLayerSize, int outputSize)
    {
        this(inputSize, hiddenLayerSize, outputSize, 1);
    }

    public NeuralNetwork(int inputSize, int hiddenLayerSize, int outputSize, int hiddenLayerCount)
    {
        // make weights and biases
        weights = new Matrix[1 + hiddenLayerCount];
        biases = new Matrix[1 + hiddenLayerCount];

        this.inputSize = inputSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.hiddenLayerCount = hiddenLayerCount;
        this.outputSize = outputSize;

        // make the network with size of input + hiddenLayerCount + output
        network = new Matrix[hiddenLayerCount + 2];

        // make network index 0 the size of input cuz it's a input layer
        network[0] = new Matrix(inputSize, 1);

        // fill network index 1 - ? with the hidden layer
        for(int i = 1; i < network.length - 1; i++)
        {
            network[i] = new Matrix(hiddenLayerSize, 1);
        }

        // make network index last the size of output cuz it's a output layer
        network[network.length - 1] = new Matrix(outputSize, 1);

        for(int i = 1; i < network.length; i++)
        {

            Matrix weight;
            Matrix bias;

            // make this if it's a hidden layer
            if(i > 1 && i != network.length - 1)
            {
                // weight from hidden to hidden
                weight = new Matrix(hiddenLayerSize, hiddenLayerSize);
            }
            else
            {
                // make this is it's a output layer
                if(i == network.length - 1)
                {
                    // weight from hidden to output
                    weight = new Matrix(outputSize, hiddenLayerSize);
                }
                // make this is it's a input layer
                else
                {
                    // weight from input to hidden
                    weight = new Matrix(hiddenLayerSize, inputSize);
                }
            }

            // fill the matrix with random values
            weight.randomize();

            // set weight array to matrix
            weights[i - 1] = weight;

            // if is not output layer make new matrix size of hidden layer else make new
            // matrix size of output layer
            if(i != network.length - 1)
            {
                bias = new Matrix(hiddenLayerSize, 1);
            }
            else
            {
                bias = new Matrix(outputSize, 1);
            }

            // randomize bias and set it to biases
            biases[i - 1] = bias;
        }

    }
    
    public void setActivationFunctions(ActivationFunction hiddenActivation, ActivationFunction outputActivation)
    {
        this.hiddenActivation = hiddenActivation;
        this.outputActivation = outputActivation;
    }

    public void setLearningRate(float learningRate)
    {
        this.learningRate = learningRate;
    }

    private void preprocess(int i, boolean train)
    {

        // matrix multiplacation between weight and before layer
        Matrix results = Matrix.multiply(weights[i - 1], network[i - 1]);

        // add biasses
        results.add(biases[i - 1]);

        if(i == network.length - 1)
        {
            ActivationFunctions.applyActivationFunction(results, outputActivation, false);
        }
        else
        {
            ActivationFunctions.applyActivationFunction(results, hiddenActivation, false);
        }

        // set current layer to the result
        network[i] = results;

        // if the current layer is the output and it's training then do the
        // backpropagation
        if(i == network.length - 1 && train)
        {
            backpropagation(results);
        }
    }

    public float[] process(float[] input)
    {
        // pass input to input layer
        network[0].fill(input);

        // feed forward the input from layer 1 so it can get layer 0
        for(int i = 1; i < network.length; i++)
        {
            preprocess(i, false);
        }

        // return the last layer aka the output
        return network[network.length - 1].toArray();
    }

    private Matrix[] getAllErrors(Matrix error)
    {
        // make array of matrix and set last index = error
        Matrix[] errors = new Matrix[weights.length];
        errors[errors.length - 1] = error;

        // calculate error on index i and pass it on index before it
        for(int i = errors.length - 1; i >= 1; i--)
        {
            errors[i - 1] = Matrix.multiply(Matrix.transpose(weights[i]), errors[i]);
        }

        return errors;
    }

    private void changingWeightsAndBiases(int index, Matrix errors, Matrix layer, Matrix afterLayer)
    {

        Matrix layerTemp = layer.clone();

        if(index == network.length - 2)
        {
            ActivationFunctions.applyActivationFunction(layerTemp, outputActivation, true);
        }
        else
        {
            ActivationFunctions.applyActivationFunction(layerTemp, hiddenActivation, true);
        }

        // multiply it by errors and learning rate
        layerTemp.scale(errors);
        layerTemp.scale(learningRate);

        // deltaWeight = gradient multiply afterLayer transposted
        Matrix deltaWeight = Matrix.multiply(layerTemp, Matrix.transpose(afterLayer));

        // adjust the weight by deltaWeight
        weights[index].add(deltaWeight);

        // adjust the bias by it's delta (it's just the gradient)
        biases[index].add(layerTemp);

    }

    private void backpropagation(Matrix output)
    {

        // calculate output error
        Matrix error = new Matrix(expectedOutput);
        error.sub(output);

        // get all errors
        Matrix[] errors = getAllErrors(error);

        // backpropagation
        for(int i = errors.length - 1; i >= 0; i--)
        {
            changingWeightsAndBiases(i, errors[i], network[i + 1], network[i]);
        }
    }

    public void train(float[] input, float[] expectedOutput)
    {
        // checking if expected output length is greater than output size
        if(expectedOutput.length > outputSize)
        {
            System.out.println("Error expected output is bigger than the output size");
            return;
        }

        // checking if input length is greater than input size
        if(expectedOutput.length > inputSize)
        {
            System.out.println("Error input is bigger than the input size");
            return;
        }

        this.expectedOutput = expectedOutput;

        // pass input to input layer
        network[0].fill(input);

        // feedforward
        for(int i = 1; i < network.length; i++)
        {
            preprocess(i, true);
        }

    }

    // mutates weights and biases by chance between 0 - 1
    public void mutate(float chance)
    {
        for(int i = 0; i < weights.length; i++)
        {
            weights[i].mutate(chance);
        }
        for(int i = 0; i < biases.length; i++)
        {
            biases[i].mutate(chance);
        }
    }

    public void crossover(NeuralNetwork other)
    {
        for(int i = 0; i < weights.length; i++)
        {

            int crossPoint = (int) (Math.random() * weights[i].cols) - 1;

            for(int j = 0; j < weights[i].rows; j++)
            {
                if(j < crossPoint)
                {
                    weights[i].array[j] = weights[i].array[j];
                }
                else
                {
                    weights[i].array[j] = other.weights[i].array[j];
                }
            }
        }

        for(int i = 0; i < biases.length; i++)
        {
            int crossPoint = (int) (Math.random() * biases[i].cols) - 1;

            for(int j = 0; j < biases[i].rows; j++)
            {
                if(j < crossPoint)
                {
                    biases[i].array[j] = biases[i].array[j];
                }
                else
                {
                    biases[i].array[j] = other.biases[i].array[j];
                }
            }
        }
    }

    public void save(String filename)
    {
        StringBuilder string = new StringBuilder("");

        for(int i = 0; i < weights.length; i++)
        {
            string.append(weights[i]);

            if(i != weights.length - 1)
                string.append('\n');
        }

        string.append("\n\n");

        for(int i = 0; i < biases.length; i++)
        {
            string.append(biases[i]);

            if(i != biases.length - 1)
                string.append('\n');
        }

        try
        {
            FileWriter writer = new FileWriter(filename);
            writer.write(string.toString());
            writer.close();
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

        System.out.println("saved to " + filename);
    }

    public static NeuralNetwork load(String filename)
    {
        String string;

        try
        {
            string = new String(Files.readAllBytes(Paths.get(filename)));
        }
        catch(IOException e)
        {
            e.printStackTrace();
            return null;
        }

        String[] elements = string.split("\n\n");
        String[] weights = elements[0].split("\n");
        String[] biasses = elements[1].split("\n");

        Matrix[] weightsMatrix = new Matrix[weights.length];
        Matrix[] biassesMatrix = new Matrix[weights.length];

        for(int i = 0; i < weights.length; i++)
        {
            weightsMatrix[i] = Matrix.fromString(weights[i]);
        }

        for(int i = 0; i < biasses.length; i++)
        {
            biassesMatrix[i] = Matrix.fromString(biasses[i]);
        }

        NeuralNetwork result = new NeuralNetwork(weightsMatrix[0].cols, weightsMatrix[0].rows, weightsMatrix[weightsMatrix.length - 1].cols, weights.length - 1);

        result.weights = weightsMatrix;
        result.biases = biassesMatrix;

        System.out.println("loaded from " + filename);

        return result;
    }

    @Override
    public NeuralNetwork clone()
    {
        NeuralNetwork clone = new NeuralNetwork(inputSize, hiddenLayerSize, outputSize, hiddenLayerCount);

        for(int i = 0; i < weights.length; i++)
        {
            clone.weights[i] = weights[i].clone();
        }
        for(int i = 0; i < biases.length; i++)
        {
            clone.biases[i] = biases[i].clone();
        }

        clone.learningRate = learningRate;
        return clone;
    }

}
