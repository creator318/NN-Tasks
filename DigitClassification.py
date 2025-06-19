import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium", app_title="")


@app.cell
def _():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    from keras import datasets, Sequential, layers, activations, optimizers, losses, metrics
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.metrics import confusion_matrix
    return (
        Sequential,
        activations,
        confusion_matrix,
        datasets,
        layers,
        losses,
        metrics,
        optimizers,
        plt,
        sns,
        tf,
    )


@app.cell
def _(datasets):
    (XTrain, yTrain), (XTest, yTest) = datasets.mnist.load_data()

    XTrain, XTest = XTrain / 255, XTest / 255.0\

    XTrain.shape, yTrain.shape, XTest.shape, yTest.shape
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
        layers.Input((28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation=activations.relu),
        layers.Dense(128, activation=activations.relu),
        layers.Dense(10, activation=activations.softmax)
    ])

    model.compile(optimizer=optimizers.Adam(), loss=losses.SparseCategoricalCrossentropy(),
                  metrics=[metrics.SparseCategoricalAccuracy()])

    _ = model.fit(XTrain, yTrain, epochs=10)
    return (model,)


@app.cell
def _(XTest, model, tf):
    yPred = tf.argmax(model.predict(XTest), axis=1)
    return (yPred,)


@app.cell
def _(confusion_matrix, plt, sns, tf, yPred, yTest):
    heatMap = sns.heatmap(
        tf.math.log1p(tf.cast(confusion_matrix(yTest, yPred), tf.float32)),
        cmap=plt.cm.coolwarm
    )
    heatMap.collections[0].colorbar.set_ticks([])

    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('Actual vs. Predicted Values')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
