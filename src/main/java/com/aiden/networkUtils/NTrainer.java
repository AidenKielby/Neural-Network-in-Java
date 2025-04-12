package com.aiden.networkUtils;

import com.aiden.Main;

import java.util.ArrayList;

public class NTrainer {
    private Network network;
    private ArrayList<ArrayList<ArrayList<Double>>> targets;
    private ArrayList<ArrayList<ArrayList<Double>>> inputs;
    private int iterations;
    private int prevPercent = -1;
    private ArrayList<Double> arrayList = new ArrayList<>();

    public NTrainer(Network network, ArrayList<ArrayList<ArrayList<Double>>> targets, ArrayList<ArrayList<ArrayList<Double>>> inputs, int iterations) {
        this.network = network;
        this.targets = targets;
        this.inputs = inputs;
        this.iterations = iterations;
    }

    public Network train() throws CloneNotSupportedException {
        int curP = 0;
        for (int i = 0; i < iterations; i++) {

            double decayRate = 0.99;
            network.setLearningRate(0.01 * Math.pow(decayRate, i / 1000.0));

            for (int trainingSample = 0; trainingSample < targets.size(); trainingSample++) {
                ArrayList<ArrayList<Double>> inputBatches = inputs.get(trainingSample);
                ArrayList<ArrayList<Double>> targetBatches = targets.get(trainingSample);

                for (int batch = 0; batch < inputBatches.size(); batch++) {
                    ArrayList<Double> inputBatch = inputBatches.get(batch);
                    ArrayList<Double> targetBatch = targetBatches.get(batch);

                    if (inputBatch.isEmpty() || targetBatch.isEmpty()) {
                        System.err.println("Skipping empty input/target pair at index " + trainingSample);
                        continue;
                    }

                    network.clearInputs();
                    network.clearTargets();

                    for (double input : inputBatch) {
                        network.addInputs(input);
                    }
                    network.setTargets(new ArrayList<>(targetBatch));

                    network.forwardPass();
                    network.backwardPass(); // This should accumulate gradients
                }
            }

            curP = (int)(((double)i / iterations) * 100);
            if (curP != prevPercent) {
                System.out.println("---------------------- \n" + curP + "%");
                prevPercent = curP;
                System.out.println("loss: " + getLoss() + "\n ----------------------");
            }

            if (i % 100 == 0) {
                System.out.println("Iteration " + i + " - Loss: " + getLoss());
            }

            arrayList.add(getLoss());
        }

        System.out.println("100%");
        System.out.println("loss: " + getLoss());
        network.clearTargets();
        network.clearInputs();
        GraphPlotter graphPlotter = new GraphPlotter(arrayList);
        graphPlotter.setVisible(true);
        return this.network;
    }

    private double getLoss(){
        return network.error();
    }
}
