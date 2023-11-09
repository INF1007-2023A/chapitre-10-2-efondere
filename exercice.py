#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def open_csv(path):
    return pandas.read_csv(path, sep=";")


def preprocess_data(df):
    return df.drop(columns="quality"), df["quality"]


def split_train_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    return (X_train, y_train), (X_test, y_test)


def forest_train(X, y):
    regr = RandomForestRegressor()
    return regr.fit(X, y)


def linear_train(X, y):
    return LinearRegression().fit(X, y)


if __name__ == "__main__":
    df = open_csv("data/winequality-white.csv")
    X, y = preprocess_data(df)
    train_set, test_set = split_train_data(X, y)

    forest_pred = forest_train(train_set[0], train_set[1])
    linear_pred = linear_train(train_set[0], train_set[1])

    forest_predictions = forest_pred.predict(test_set[0])
    linear_predictions = linear_pred.predict(test_set[0])

    print(
        f"Mean square error (forest): {mean_squared_error(test_set[1], forest_predictions)}"
    )
    print(
        f"Mean square error (linear): {mean_squared_error(test_set[1], linear_predictions)}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(test_set[1].to_numpy(), label="Target values")
    axes[0].plot(forest_predictions, c="darkorange", label="Predicted values")
    axes[0].set_title("RandomForestRegressor prediction analysis")
    axes[0].set(xlabel="Number of samples")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim([2.8, 9.2])

    axes[1].plot(test_set[1].to_numpy(), label="Target values")
    axes[1].plot(linear_predictions, c="darkorange", label="Predicted values")
    axes[1].set_title("LinearRegression prediction analysis")
    axes[1].set(xlabel="Number of samples")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim([2.8, 9.2])

    for ax in axes.flat:
        ax.set(ylabel="Quality")
        ax.label_outer()

    plt.show()
