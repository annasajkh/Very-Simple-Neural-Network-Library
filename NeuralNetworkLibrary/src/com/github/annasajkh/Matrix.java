package com.github.annasajkh;

import java.util.Arrays;


public class Matrix
{
    double[][] array;
    int rows;
    int cols;

    public Matrix(int rows, int cols)
    {
        this.rows = rows;
        this.cols = cols;

        array = new double[rows][cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i][j] = 0;
            }
        }
    }

    public Matrix(double[] arr)
    {
        this.rows = arr.length;
        this.cols = 1;

        array = new double[this.rows][this.cols];
        for (int i = 0; i < this.rows; i++)
        {
            array[i][0] = arr[i];
        }

    }


    public void randomize()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    public void scale(double scalar)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i][j] *= scalar;
            }
        }
    }

    public void scale(Matrix matrix)
    {
        for (int i = 0; i < matrix.rows; i++)
        {
            for (int j = 0; j < matrix.cols; j++)
            {
                array[i][j] *= matrix.array[i][j];
            }
        }
    }

    public void mutate(double chance)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i][j] += Math.random() <= chance ? Math.random() * 2 - 1: 0;
            }
        }
    }

    public double[] toArray()
    {
        double[] result = new double[rows * cols];
        int index = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[index] = array[i][j];
                index++;
            }
        }
        return result;
    }

    public static Matrix transpose(Matrix matrix)
    {
        Matrix result = new Matrix(matrix.cols, matrix.rows);
        for (int i = 0; i < matrix.rows; i++)
        {
            for (int j = 0; j < matrix.cols; j++)
            {
                result.array[j][i] = matrix.array[i][j];
            }
        }
        return result;

    }

    public static Matrix multiply(Matrix a, Matrix b)
    {
        if (a.cols != b.rows)
        {
            System.out.println("columns has match to rows");
            return null;
        }
        Matrix result = new Matrix(a.rows, b.cols);
        for (int i = 0; i < result.rows; i++)
        {
            for (int j = 0; j < result.cols; j++)
            {
                double sum = 0;
                for (int k = 0; k < a.cols; k++)
                {
                    sum += a.array[i][k] * b.array[k][j];
                }
                result.array[i][j] = sum;
            }
        }
        return result;
    }


    public void add(Matrix matrix)
    {
        for (int i = 0; i < matrix.rows; i++)
        {
            for (int j = 0; j < matrix.cols; j++)
            {
                array[i][j] += matrix.array[i][j];
            }
        }
    }

    public void sub(Matrix matrix)
    {
        for (int i = 0; i < matrix.rows; i++)
        {
            for (int j = 0; j < matrix.cols; j++)
            {
                array[i][j] -= matrix.array[i][j];
            }
        }
    }

    public void add(double scalar)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i][j] += scalar;
            }
        }
    }

    public Matrix clone()
    {
        Matrix matrix = new Matrix(this.rows,this.cols);
        for (int i = 0; i < matrix.rows; i++)
        {
            for (int j = 0; j < matrix.cols; j++)
            {
                matrix.array[i][j] = array[i][j];
            }
        }
        return matrix;
    }


    @Override
    public String toString()
    {
        return Arrays.deepToString(array);
    }
}
