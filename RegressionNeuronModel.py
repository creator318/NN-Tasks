import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras import Sequential, layers, activations, optimizers, losses, metrics
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    return (
        Sequential,
        activations,
        layers,
        losses,
        make_regression,
        metrics,
        optimizers,
        plt,
        train_test_split,
    )


@app.cell
def _(make_regression, train_test_split):
    X, y = make_regression(n_samples=2000, n_features=2, noise=20, random_state=0)

    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return XTest, XTrain, yTest, yTrain


@app.cell
def _(
    Sequential,
    XTrain,
    activations,
    layers,
    losses,
    metrics,
    optimizers,
    yTrain,
):
    model = Sequential([
        layers.Input((2,)),
        layers.Dense(1, activation=activations.linear)
    ])

    model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(),
                  metrics=[metrics.R2Score(), metrics.MeanAbsoluteError()])

    _ = model.fit(XTrain, yTrain, epochs=50)
    return (model,)


@app.cell
def _(XTest, model):
    yPred = model.predict(XTest)
    return (yPred,)


@app.cell
def _(plt, yPred, yTest):
    plt.scatter(x=yTest, y=yPred, alpha=0.3, s=20, linewidth=0)
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Actual vs. Predicted Values')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
