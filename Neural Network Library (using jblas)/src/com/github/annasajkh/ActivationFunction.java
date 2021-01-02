package com.github.annasajkh;

public enum ActivationFunction
{
    //use sigmoid if you want standard thing is not fast or slow and it's accurate
    SIGMOID,
    /*
    use tanh if you want more accurate but be careful if you change the learning rate to high or
    make training too long it may overshot
    */
    TANH,

    //use leaky_relu if you want very high speed but it's sometimes unpredictable
    LEAKY_RELU,
}
