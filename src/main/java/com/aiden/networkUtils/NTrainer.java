package com.aiden.networkUtils;

import com.aiden.Main;

import java.util.ArrayList;

public class NTrainer {
    private Network network;
    private ArrayList<ArrayList<Double>> targets;
    private ArrayList<ArrayList<Double>> inputs;
    private int iterations;
    private int prevPercent = -1;
    private ArrayList<Double> arrayList = new ArrayList<>();

    public NTrainer(Network network, ArrayList<ArrayList<Double>> targets, ArrayList<ArrayList<Double>> inputs, int iterations) {
        this.network = network;
        this.targets = targets;
        this.inputs = inputs;
        this.iterations = iterations;
    }

    public Network train(){
        int curP = 0;
        for (int i = 0; i < iterations; i++) {

            network.setLearningRate(0.01 / (1 + i / (double) iterations));

            for (int trainingSample = 0; trainingSample < targets.size(); trainingSample++) {
                ArrayList<Double> ins = inputs.get(trainingSample);
                ArrayList<Double> targs = targets.get(trainingSample);

                // Check if inputs or targets are empty before proceeding
                if (!(ins.isEmpty() || targs.isEmpty())) {
                    network.clearInputs();
                    network.clearTargets();

                    for (double input : ins){
                        network.addInputs(input);
                    }
                    network.setTargets(new ArrayList<>(targs));

                    // System.out.println(targs);
                    // System.out.println(!(ins.isEmpty() || targs.isEmpty()));
                    // System.out.println(ins);

                    network.forwardPass();
                    network.backwardPass();
                }
                else {
                    System.err.println("Skipping empty input/target pair at index " + trainingSample);
                }
            }

            curP = (int)(((double)i/iterations)*100);
            if (curP != prevPercent){
                System.out.println("---------------------- \n" + curP + "%");
                prevPercent = curP;
                System.out.println("loss: " + getLoss() + "\n ----------------------");
            }

            if (i % 100 == 0) {
                System.out.println("Iteration " + i + " - Loss: " + getLoss());
            }

            if (Network.roundToDecimal(getLoss(), 1) < 0.5){
                System.out.println("100%");
                System.out.println("loss: " + getLoss());
                network.clearTargets();
                network.clearInputs();
                GraphPlotter graphPlotter = new GraphPlotter(arrayList);
                graphPlotter.setVisible(true);
                return this.network;
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
