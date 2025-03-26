package com.aiden.networkUtils;

import java.io.*;

public class SaveOrLoad {
    public SaveOrLoad() {
    }

    public static void saveNetwork(Network network, String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(network);
            System.out.println("Network saved successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static Network loadNetwork(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return (Network) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
}
