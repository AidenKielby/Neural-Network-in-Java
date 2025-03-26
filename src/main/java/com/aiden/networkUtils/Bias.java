package com.aiden.networkUtils;

import java.io.Serializable;

public class Bias implements Serializable {
    private double bias = 0;
    private int neuronIndex;

    public Bias(double bias, int neuronIndex) {
        this.bias = bias;
        this.neuronIndex = neuronIndex;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public int getNeuronIndex() {
        return neuronIndex;
    }

    public void setNeuronIndex(int neuronIndex) {
        this.neuronIndex = neuronIndex;
    }
}
