package com.aiden.networkUtils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.ArrayList;

public class GraphPlotter extends JFrame {
    public GraphPlotter(ArrayList<Double> data) {
        setTitle("Graph Plot");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        XYSeries series = new XYSeries("Data Points");
        for (int i = 0; i < data.size(); i++) {
            series.add(i, data.get(i));  // X = index, Y = value
        }

        XYSeriesCollection dataset = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Data Visualization",
                "Index",
                "Value",
                dataset
        );

        ChartPanel panel = new ChartPanel(chart);
        setContentPane(panel);
    }

    public static void main(String[] args) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            data.add(Math.sin(i * 0.1) * 10);  // Example data (sine wave)
        }

        SwingUtilities.invokeLater(() -> {
            GraphPlotter plotter = new GraphPlotter(data);
            plotter.setVisible(true);
        });
    }
}
