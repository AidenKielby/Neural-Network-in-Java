package com.aiden.networkUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

public class Network implements Serializable {
    private ArrayList<Neuron> neurons;
    private ArrayList<Weight> weights;
    private ArrayList<Bias> biases;
    private final int neuronsInInputLayer;
    private final int hiddenLayers;
    private final int neuronsPerHiddenLayer;
    private final int neuronsInOutputLayer;
    private ArrayList<Double> inputs;
    private ArrayList<Double> targets;
    private ArrayList<Double> outputLayerError;
    private ArrayList<ArrayList<Double>> hiddenLayersError;
    private double learningRate;
    private String activationFunction;
    private static final double ALPHA = 0.01;

    /**
     * Initializes the network, activation function can be either Leaky ReLU, Sigmoid or ReLU
     *
     * @param activationFunction the input string (e.g., "L-ReLU", "Sigmoid", or "ReLU")
     */
    public Network(int neuronsInInputLayer, int hiddenLayers, int neuronsPerHiddenLayer, int neuronsInOutputLayer, String activationFunction, double learningRate) {
        this.activationFunction = activationFunction;
        this.neuronsInInputLayer = neuronsInInputLayer;
        this.hiddenLayers = hiddenLayers;
        this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;
        this.neuronsInOutputLayer = neuronsInOutputLayer;

        neurons = new ArrayList<>();
        addNeurons();
        weights = new ArrayList<>();
        addWeights();
        biases = new ArrayList<>();
        addBiases();

        this.inputs = new ArrayList<>();
        outputLayerError = new ArrayList<>();
        hiddenLayersError = new ArrayList<>();
        targets = new ArrayList<>();
        this.learningRate = learningRate;
    }

    private double HeInit(int in, int out) {
        return Math.sqrt(((double) 2 /(in + out)));
    }

    public static double getRandomInRange(double x) {
        Random random = new Random();
        return random.nextGaussian() * x;
    }

    public static double roundToDecimal(double value, int places) {
        double scale = Math.pow(10, places);
        return Math.round(value * scale) / scale;
    }

    public ArrayList<Double> getInputs() {
        return inputs;
    }

    public void addInputs(double input) {
        this.inputs.add(input);
    }

    public void clearInputs(){this.inputs.clear();}

    public void clearTargets(){targets.clear();}

    public void setTargets(ArrayList<Double> targets) {
        this.targets = targets;
    }

    // -----------------------
    private void addNeurons(){
        for (int inputN = 0; inputN < neuronsInInputLayer; inputN++){
            neurons.add(new Neuron(0));
        }

        for (int layers = 0; layers < hiddenLayers; layers++) {
            for (int hiddenN = 0; hiddenN < neuronsPerHiddenLayer; hiddenN++) {
                neurons.add(new Neuron(1));
            }
        }

        for (int outputN = 0; outputN < neuronsInOutputLayer; outputN++) {
            neurons.add(new Neuron(2));
        }
    }

    private void addWeights(){
        for (int inputN = 0; inputN < neuronsInInputLayer; inputN++) {
            for (int middleN = 0; middleN < neuronsPerHiddenLayer; middleN++) {
                weights.add(new Weight(inputN, middleN+neuronsInInputLayer, getRandomInRange(HeInit(neuronsInInputLayer, neuronsPerHiddenLayer))));
            }
        }

        for (int hiddenL = 0; hiddenL < hiddenLayers - 1; hiddenL++) {
            for (int hiddenN = 0; hiddenN < neuronsPerHiddenLayer; hiddenN++) {
                for (int hiddenN1 = 0; hiddenN1 < neuronsPerHiddenLayer; hiddenN1++) {
                    weights.add(new Weight(hiddenN+(hiddenL*neuronsPerHiddenLayer)+neuronsInInputLayer, hiddenN1+(hiddenL*neuronsPerHiddenLayer+neuronsPerHiddenLayer)+neuronsInInputLayer, getRandomInRange(HeInit(neuronsPerHiddenLayer, neuronsPerHiddenLayer))));
                }
            }
        }

        for (int hiddenN = 0; hiddenN < neuronsPerHiddenLayer; hiddenN++) {
            for (int outputN = 0; outputN < neuronsInOutputLayer; outputN++) {
                weights.add(new Weight(hiddenN+neuronsInInputLayer+((hiddenLayers-1)*neuronsPerHiddenLayer), outputN+(neuronsPerHiddenLayer*hiddenLayers)+neuronsInInputLayer, getRandomInRange(HeInit(neuronsPerHiddenLayer, neuronsInOutputLayer))));
            }
        }

    }

    private void addBiases(){
        for (int neuron = 0; neuron < neurons.size(); neuron++) {
            if (neurons.get(neuron).getType() != 0){
                biases.add(new Bias(Math.random(), neuron));
            }
        }
    }

    public void setLearningRate(double learningRate){
        this.learningRate = learningRate;
    }

    // -----------------------
    public void forwardPass(){
        clearNeuronValues();
        loopThroughHiddenLayers();
        loopThroughOutputLayers();
    }

    public void clearNeuronValues(){
        for (int neuronIndex = 0; neuronIndex < neurons.size(); neuronIndex++) {
            if (neuronIndex < neuronsInInputLayer) {
                neurons.get(neuronIndex).setActivation(inputs.get(neuronIndex));
            }
            else{
                neurons.get(neuronIndex).setValue(0);
            }
        }
    }

    private void loopThroughHiddenLayers(){
        for (int hiddenLayerIndex = 0; hiddenLayerIndex < hiddenLayers; hiddenLayerIndex++) {

            for (int neuronIndex = 0; neuronIndex < neuronsPerHiddenLayer; neuronIndex++) {
                Neuron neuron = neurons.get((neuronsPerHiddenLayer*hiddenLayerIndex+neuronIndex) + neuronsInInputLayer);
                for (Weight weight : weights){

                    if (weight.getNeuronIndex2() == (neuronsPerHiddenLayer*hiddenLayerIndex+neuronIndex) + neuronsInInputLayer){
                        Neuron fromNeuron = neurons.get(weight.getNeuronIndex1());
                        Neuron toNeuron = neurons.get(weight.getNeuronIndex2());

                        toNeuron.setValue(toNeuron.getActivation() + (fromNeuron.getActivation() * weight.getWeight()));

                    }

                }

                for (Bias bias : biases){

                    if (bias.getNeuronIndex() == (neuronsPerHiddenLayer*hiddenLayerIndex+neuronIndex) + neuronsInInputLayer){

                        neuron.setValue(neuron.getActivation() + bias.getBias());
                    }

                }
                if (Objects.equals(activationFunction, "Sigmoid")){
                    neuron.setValue(sigmoid(neuron.getActivation()));
                } else if (Objects.equals(activationFunction, "L-ReLU")) {
                    neuron.setValue(roundToDecimal(leakyReLU(neuron.getActivation()), 10));
                } else if (Objects.equals(activationFunction, "ReLU")) {
                    neuron.setValue(roundToDecimal(ReLU(neuron.getActivation()), 10));
                }
            }

        }
    }

    private void loopThroughOutputLayers(){
        for (int neuronIndex = 0; neuronIndex < neuronsInOutputLayer; neuronIndex++) {
            Neuron neuron = neurons.get(neuronIndex + (neuronsPerHiddenLayer * hiddenLayers) + neuronsInInputLayer);

            for (Weight weight : weights){

                if (weight.getNeuronIndex2() == neuronIndex + (neuronsPerHiddenLayer * hiddenLayers) + neuronsInInputLayer){
                    Neuron fromNeuron = neurons.get(weight.getNeuronIndex1());
                    Neuron toNeuron = neurons.get(weight.getNeuronIndex2());

                    toNeuron.setValue(toNeuron.getActivation() + (fromNeuron.getActivation() * weight.getWeight()));

                }

            }

            for (Bias bias : biases){

                if (bias.getNeuronIndex() == neuronIndex + (neuronsPerHiddenLayer * hiddenLayers) + neuronsInInputLayer){

                    neuron.setValue(neuron.getActivation() + bias.getBias());
                }

            }
            if (Objects.equals(activationFunction, "Sigmoid")){
                neuron.setValue(sigmoid(neuron.getActivation()));
            } else if (Objects.equals(activationFunction, "L-ReLU")) {
                neuron.setValue(leakyReLU(neuron.getActivation()));
            }

        }
    }

    public static double leakyReLU(double x) {
        double alpha = 0.01;
        return Math.max(alpha * x, x);
    }

    public double leakyReLUDerivativeFromOutput(double output) {
        double alpha = 0.01;  // Keep α small
        return (output >= 0) ? 1.0 : alpha;
    }

    private double ReLU(double x){
        return Math.max(0, x);
    }

    private double ReLUDerivative(double x){
        return (x >= 0) ? 1.0 : 0;
    }

    public static double sigmoidDerivativeFromOutput(double output) {
        return output * (1 - output);
    }

    private double sigmoid(double x){
        return 1/(1 + Math.exp(-x));
    }

    public static double[] softmax(double[] inputs) {
        double max = Arrays.stream(inputs).max().orElse(0); // Avoid large exponentials
        double sum = 0;
        double[] expValues = new double[inputs.length];

        // Compute e^(input - max) for numerical stability
        for (int i = 0; i < inputs.length; i++) {
            expValues[i] = Math.exp(inputs[i] - max);
            sum += expValues[i];
        }

        // Normalize to get probabilities
        for (int i = 0; i < inputs.length; i++) {
            expValues[i] /= sum;
        }

        return expValues;
    }

    // -----------------------
    public double error(){
        return calculateError();
    }

    public void backwardPass(){
        double totalLoss = calculateError();
        findOutputLayerError();
        findMiddleLayersError();
        updateWeights();
        updateBiases();
    }

    private double calculateError(){
        double loss = 0;
        for (int i = 0; i < neuronsInOutputLayer; i++) {
            double predicted = neurons.get(neuronsInInputLayer + (neuronsPerHiddenLayer * hiddenLayers) + i).getActivation();
            double target = targets.get(i);
            loss -= target * Math.log(predicted + 1e-9);  // Add 1e-9 to avoid log(0)
        }
        return loss;
    }

    private void findOutputLayerError(){
        outputLayerError.clear();

        for (int outputNeuronIndex = neuronsInInputLayer + (neuronsPerHiddenLayer * hiddenLayers); outputNeuronIndex < neurons.size(); outputNeuronIndex++) {
            Neuron neuron = neurons.get(outputNeuronIndex);
            double neuronValue = neuron.getActivation();
            double target = targets.get(outputNeuronIndex - (neuronsInInputLayer + (neuronsPerHiddenLayer * hiddenLayers)));

            if (Objects.equals(activationFunction, "Sigmoid")){
                double currentNeuronLoss = (neuronValue - target) * sigmoidDerivativeFromOutput(neuronValue);
                outputLayerError.add(currentNeuronLoss);
            }
            else if (Objects.equals(activationFunction, "L-ReLU")){
                double currentNeuronLoss = (neuronValue - target) * leakyReLUDerivativeFromOutput(neuronValue);
                outputLayerError.add(currentNeuronLoss);
            } else if (Objects.equals(activationFunction, "ReLU")){
                double currentNeuronLoss = (neuronValue - target) * ReLUDerivative(neuronValue);
                outputLayerError.add(currentNeuronLoss);
            }


        }
    }

    private void findMiddleLayersError(){
        hiddenLayersError.clear();

        for (int hiddenLayer = hiddenLayers-1; hiddenLayer > -1; hiddenLayer--) {
            ArrayList<Double> hiddenLayerError = new ArrayList<>();

            if (hiddenLayer == hiddenLayers-1) {

                for (int neuronIndex = 0; neuronIndex < neuronsPerHiddenLayer; neuronIndex++) {
                    Neuron neuron = neurons.get((neuronsPerHiddenLayer * hiddenLayer + neuronIndex) + neuronsInInputLayer);
                    double neuronValue = neuron.getActivation();

                    double summed = sumNeuronContributions((neuronsPerHiddenLayer * hiddenLayer + neuronIndex) + neuronsInInputLayer);

                    if (Objects.equals(activationFunction, "Sigmoid")){
                        double currentNeuronLoss = summed * sigmoidDerivativeFromOutput(neuronValue);
                        hiddenLayerError.add(currentNeuronLoss);
                    }
                    else if (Objects.equals(activationFunction, "L-ReLU")){
                        double currentNeuronLoss = summed * leakyReLUDerivativeFromOutput(neuronValue);
                        hiddenLayerError.add(currentNeuronLoss);
                    } else if (Objects.equals(activationFunction, "ReLU")){
                        double currentNeuronLoss = summed * ReLUDerivative(neuronValue);
                        hiddenLayerError.add(currentNeuronLoss);
                    }




                }
            }
            else {
                for (int neuronIndex = 0; neuronIndex < neuronsPerHiddenLayer; neuronIndex++) {
                    Neuron neuron = neurons.get((neuronsPerHiddenLayer * hiddenLayer + neuronIndex) + neuronsInInputLayer);
                    double neuronValue = neuron.getActivation();

                    double summed = sumHiddenNeuronContributions((neuronsPerHiddenLayer * hiddenLayer + neuronIndex) + neuronsInInputLayer, hiddenLayer+1);

                    if (Objects.equals(activationFunction, "Sigmoid")){
                        double currentNeuronLoss = summed * sigmoidDerivativeFromOutput(neuronValue);
                        hiddenLayerError.add(currentNeuronLoss);
                    }
                    else if (Objects.equals(activationFunction, "L-ReLU")){
                        double currentNeuronLoss = summed * leakyReLUDerivativeFromOutput(neuronValue);
                        currentNeuronLoss = roundToDecimal(currentNeuronLoss, 10);
                        hiddenLayerError.add(currentNeuronLoss);
                    } else if (Objects.equals(activationFunction, "ReLU")){
                        double currentNeuronLoss = summed * ReLUDerivative(neuronValue);
                        currentNeuronLoss = roundToDecimal(currentNeuronLoss, 10);
                        hiddenLayerError.add(currentNeuronLoss);
                    }
                }
            }

            hiddenLayersError.add(hiddenLayerError);

        }
    }

    private double sumNeuronContributions(int index){
        double thing = 0;
        for (int outputNeuronIndex = 0; outputNeuronIndex < neuronsInOutputLayer; outputNeuronIndex++) {
            double errorTerm = outputLayerError.get(outputNeuronIndex);
            for (Weight weight : weights){

                if (weight.getNeuronIndex2() == (neuronsPerHiddenLayer*hiddenLayers) + neuronsInInputLayer+outputNeuronIndex &&
                weight.getNeuronIndex1() == index){
                    thing += weight.getWeight()*errorTerm;

                }

            }
        }
        if (Double.isNaN(thing)) {
            System.err.println("Summed neuron contribution is NaN!");
        }
        return thing;
    }

    private double sumHiddenNeuronContributions(int index, int layerIndex){
        double thing = 0;
        for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < neuronsPerHiddenLayer; hiddenNeuronIndex++) {
            double errorTerm = hiddenLayersError.getLast().get(hiddenNeuronIndex);
            for (Weight weight : weights){

                if (weight.getNeuronIndex2() == (neuronsPerHiddenLayer * (layerIndex) + hiddenNeuronIndex) + neuronsInInputLayer &&
                        weight.getNeuronIndex1() == index){
                    thing += weight.getWeight()*errorTerm;

                }

            }
        }
        return thing;
    }

    private void updateWeights(){
         for (Weight weight : weights){
             Neuron prevNeuron = neurons.get(weight.getNeuronIndex1());
             Neuron currentNeuron = neurons.get(weight.getNeuronIndex2());
             double gradient = 0;

             if (isNeuronInOutputLayer(weight.getNeuronIndex2())){
                 int outputLayerErrorIndex = weight.getNeuronIndex2() - (neuronsInInputLayer + hiddenLayers * neuronsPerHiddenLayer);
                 double error = outputLayerError.get(outputLayerErrorIndex);
                 gradient = error * prevNeuron.getActivation();

             }
             else if (isNeuronInMiddleLayer(weight.getNeuronIndex2())) {
                 ArrayList<Double> list = newMiddleList();
                 int hiddenLayerErrorIndex = weight.getNeuronIndex2() - (neuronsInInputLayer);
                 double error = list.get(hiddenLayerErrorIndex);
                 gradient = error * prevNeuron.getActivation();
             }

             double gradient1 = gradient;
             double clipValue = 5.0;  // Prevent too-large updates
             if (gradient1 > clipValue) gradient1 = clipValue;
             if (gradient1 < -clipValue) gradient1 = -clipValue;

             double weightVal = weight.getWeight();
             double m = 0, v = 0; // Moving averages
             double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
             m = beta1 * m + (1 - beta1) * gradient;
             v = beta2 * v + (1 - beta2) * gradient * gradient;
             double mHat = m / (1 - beta1);
             double vHat = v / (1 - beta2);
             weight.setWeight(weightVal - learningRate * mHat / (Math.sqrt(vHat) + epsilon));

         }
    }

    private void updateBiases(){
        for (Bias bias : biases){
            if (isNeuronInOutputLayer(bias.getNeuronIndex())){
                int outputLayerErrorIndex = bias.getNeuronIndex() - (neuronsInInputLayer + hiddenLayers * neuronsPerHiddenLayer);
                double error = outputLayerError.get(outputLayerErrorIndex);
                bias.setBias(error);
            }
            else if (isNeuronInMiddleLayer(bias.getNeuronIndex())){
                ArrayList<Double> list = newMiddleList();
                int hiddenLayerErrorIndex = bias.getNeuronIndex() - (neuronsInInputLayer);
                double error = list.get(hiddenLayerErrorIndex);
                bias.setBias(error);
            }
        }
    }

    private ArrayList<Double> newMiddleList(){
        ArrayList<Double> list = new ArrayList<>();

        for (int index = hiddenLayersError.size(); index > 0; index--) {
            ArrayList<Double> doubles = hiddenLayersError.get(index-1);
            list.addAll(doubles);
        }

        return list;
    }
    // -----------------------

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();

        for (int outputNeuronIndex = 0; outputNeuronIndex < neuronsInOutputLayer; outputNeuronIndex++) {
            stringBuilder.append(neurons.get(neuronsInInputLayer + (neuronsPerHiddenLayer * hiddenLayers) + outputNeuronIndex).getActivation());

            if (outputNeuronIndex < neuronsInOutputLayer-1){
                stringBuilder.append(", ");
            }
        }

        return stringBuilder.toString();
    }

    public ArrayList<Double> toList() {
        ArrayList<Double> arrayList = new ArrayList<>();

        for (int outputNeuronIndex = 0; outputNeuronIndex < neuronsInOutputLayer; outputNeuronIndex++) {
            arrayList.add(neurons.get(neuronsInInputLayer + (neuronsPerHiddenLayer * hiddenLayers) + outputNeuronIndex).getActivation());
        }

        return arrayList;
    }

    private boolean isNeuronInOutputLayer(int neuronIndex){
        return neuronIndex+1 > neuronsInInputLayer + neuronsPerHiddenLayer * hiddenLayers;
    }

    private boolean isNeuronInMiddleLayer(int neuronIndex){
        return neuronIndex+1 > neuronsInInputLayer && neuronIndex < neuronsInInputLayer + neuronsPerHiddenLayer * hiddenLayers;
    }

    private void printWeights(){
        for (Weight weight : weights){
            System.out.println(weight.getWeight());
        }
    }

    public int returnLargestOutputInFormOfIndex(){
        double max = -999;
        int maxIndex = 0;
        for (int i = 0; i < neuronsInOutputLayer; i++) {
            Neuron neuron = neurons.get(i + neuronsInInputLayer + (hiddenLayers * neuronsPerHiddenLayer));
            if (max < neuron.getActivation()){max = neuron.getActivation(); maxIndex = i;}
        }
        return maxIndex;
    }

    @Override
    protected Object clone() throws CloneNotSupportedException {
        return new Network(neuronsInInputLayer, hiddenLayers, neuronsPerHiddenLayer, neuronsInOutputLayer, activationFunction, learningRate);
    }

    private double inputLayerMean(){
        double mean = 0;

        for (int i = 0; i < neuronsInInputLayer; i++) {
            mean += neurons.get(i).getActivation();
        }

        mean/=neuronsInInputLayer;
        return mean;
    }

    private double standardDeviationOfInputLayer(double mean){
        double sD = 0;

        for (int i = 0; i < neuronsInInputLayer; i++) {
            double diff = neurons.get(i).getActivation() - mean;
            sD += diff * diff;
        }

        sD/=neuronsInInputLayer;
        return sD;
    }

    public void zScoreNormalizeInputLayer(){
        double mean = inputLayerMean();
        for (int i = 0; i < neuronsInInputLayer; i++) {
            double norm = ((neurons.get(i).getActivation() - mean)/standardDeviationOfInputLayer(mean));
            neurons.get(i).setValue(norm);
        }
    }

}
