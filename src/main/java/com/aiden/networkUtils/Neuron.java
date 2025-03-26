package com.aiden.networkUtils;

import java.io.Serializable;

public class Neuron implements Serializable {
    private double activation = 0;
    // type = 0 if input layer, type = 1 if hidden layer, type = 2 if output layer
    private int type;

    public Neuron(int type) {
        this.type = type;
    }

    public Neuron() {
        this.activation = 1;
    }

    public double getActivation() {
        if (Double.isNaN(activation)) {
            System.err.println("Activation is NaN!");
        }
        return activation;
    }

    public void setActivation(double activation) {
        if (this.type == 0) {
            this.activation = activation;
        }
    }

    public void setValue(double value){
        this.activation = value;
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }
}
