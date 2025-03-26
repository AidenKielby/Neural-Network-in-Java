package com.aiden.networkUtils;

import java.io.Serializable;

public class Weight implements Serializable {
    private double weight = 1;
    private int neuronIndex1;
    private int neuronIndex2;

    public Weight(int neuronIndex1, int neuronIndex2, double weight) {
        this.neuronIndex1 = neuronIndex1;
        this.neuronIndex2 = neuronIndex2;
        this.weight = weight;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public int getNeuronIndex1() {
        return neuronIndex1;
    }

    public void setNeuronIndex1(int neuronIndex1) {
        this.neuronIndex1 = neuronIndex1;
    }

    public int getNeuronIndex2() {
        return neuronIndex2;
    }

    public void setNeuronIndex2(int neuronIndex2) {
        this.neuronIndex2 = neuronIndex2;
    }
}
