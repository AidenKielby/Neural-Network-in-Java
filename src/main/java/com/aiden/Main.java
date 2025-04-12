package com.aiden;

import com.aiden.networkUtils.NTrainer;
import com.aiden.networkUtils.Network;
import com.aiden.networkUtils.SaveOrLoad;

import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws CloneNotSupportedException {
        Random random = new Random();
        Scanner scanner = new Scanner(System.in);  // Create a Scanner object
        System.out.println("Would you like to load a network? (Y/N)");

        String answer = scanner.nextLine();
        if (Objects.equals(answer, "Y") || Objects.equals(answer, "y")){
            System.out.println("What is the filename?");

            String filename = scanner.nextLine();
            Network network = SaveOrLoad.loadNetwork(filename);

            System.out.println("How many inputs?");
            String inputs = scanner.nextLine();

            for (int i = 0; i < Integer.parseInt(inputs); i++) {
                System.out.println("What is this input");
                double input = Double.parseDouble(scanner.nextLine());

                assert network != null;
                network.addInputs(input);
            }

            assert network != null;
            network.forwardPass();
            System.out.println(network.toList());
            if (network.returnLargestOutputInFormOfIndex() == 0){
                System.out.println("Network says the point was over the line!");
            }
            else{
                System.out.println("Network says the point was under the line!");
            }
        }

        else{
            Network network = new Network(2, 3, 5, 2, "L-ReLU", 0.00001); //uses He init
            ArrayList<ArrayList<Double>> inputs = new ArrayList<>();
            ArrayList<ArrayList<Double>> targets = new ArrayList<>();
            for (int i = 0; i < 1000; i++) {
                double x = Network.roundToDecimal(random.nextDouble(999), 0);
                double y;
                int aboveOrBellow = random.nextInt(2);
                ArrayList<Double> target = new ArrayList<>();
                if (aboveOrBellow == 0){
                    y = Network.roundToDecimal(x + random.nextDouble() * 15, 0);  // Above the line
                    target.add(1.0);
                    target.add(0.0);
                } else {
                    y = Network.roundToDecimal(x - random.nextDouble() * 15, 0);  // Below the line
                    target.add(0.0);
                    target.add(1.0);
                }

                ArrayList<Double> input = new ArrayList<>();
                input.add((x - 499.5) / 499.5);
                input.add((y - 499.5) / 499.5);
                System.out.println((x - 499.5) / 499.5 + " " + (y - 499.5) / 499.5);

                inputs.add(input);
                targets.add(target);
            }

            NTrainer networkTrainer = new NTrainer(network, targets, inputs, 10_000);
            Network network1 = networkTrainer.train();

            int correct = 0;

            for (int iteration = 0; iteration < 1000; iteration++) {
                double x = Network.roundToDecimal(random.nextDouble(999), 0);
                double y;
                int resultIndex;
                int aboveOrBellow = random.nextInt(2);
                if (aboveOrBellow == 0){
                    y = Network.roundToDecimal(x + random.nextDouble() * 15, 0);
                    resultIndex = 0;
                } else {
                    y = Network.roundToDecimal(x - random.nextDouble() * 15, 0);
                    resultIndex = 1;
                }

                network1.clearInputs();
                network1.addInputs(x);
                network1.addInputs(y);

                network1.forwardPass();
                if (network1.returnLargestOutputInFormOfIndex() == resultIndex) {
                    //System.out.println("YAY, It was correct!");
                    correct ++;
                }
                else {
                    System.out.println("wrong :(");
                    System.out.println(x);
                    System.out.println(y);
                }

            }

            double percent = (correct / 1000.0) * 100;
            System.out.println(percent + "% correct");

            System.out.println("Would you like to save this network? (Y/N)");

            answer = scanner.nextLine();

            if (Objects.equals(answer, "Y") || Objects.equals(answer, "y")){
                System.out.println("What will the filename be?");

                String filename = scanner.nextLine();
                SaveOrLoad.saveNetwork(network1, filename);
            }
        }
    }
}