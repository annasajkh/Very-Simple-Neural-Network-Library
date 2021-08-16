package com.github.annasajkh;

import java.util.function.UnaryOperator;

class ActivationFunction
{
    UnaryOperator<Float> activationFunction;
    UnaryOperator<Float> derivative;
    
    public ActivationFunction(UnaryOperator<Float> activationFunction, UnaryOperator<Float> derivative)
    {
        this.activationFunction = activationFunction;
        this.derivative = derivative;
    }
    
}

public class ActivationFunctions
{
    public static ActivationFunction sigmoid = new ActivationFunction(x -> 1.0f / (1.0f + (float) (Math.exp(-x))), 
                                                                      x -> x * (1 - x));
    
    public static ActivationFunction leakyRelu = new ActivationFunction(x -> x >= 0 ? x : 0.01f * x,
                                                                        x -> x >= 0 ? 1 : 0.01f);
    
    public static ActivationFunction tanh = new ActivationFunction(x -> (float) (Math.exp(x) - Math.exp(-x) / Math.exp(x) + Math.exp(-1)),
                                                                   x -> 1 - (float) (Math.pow(x, 2)));

    public static void applyActivationFunction(Matrix matrix, ActivationFunction function, boolean isDerivative)
    {
        for(int i = 0; i < matrix.rows; i++)
        {
            for(int j = 0; j < matrix.cols; j++)
            {
                if(isDerivative)
                {
                    matrix.array[i][j] = function.derivative.apply(matrix.array[i][j]);
                }
                else
                {
                    matrix.array[i][j] = function.activationFunction.apply(matrix.array[i][j]);
                }
            }
        }
    }

}
