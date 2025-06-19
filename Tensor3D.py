import marimo

__generated_with = "0.14.0"
app = marimo.App()


@app.cell
def _():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    return (tf,)


@app.cell
def _(tf):
    tf.random.set_seed(0)

    t1 = tf.random.uniform((2, 3, 4), 1, 10)
    t2 = tf.random.uniform((2, 3, 4), 1, 10)
    return t1, t2


@app.cell
def _(t1, t2):
    print("Tensor 1:", t1, end="\n\n")
    print("Tensor 2:", t2, end="\n\n")
    return


@app.cell
def _(t1, t2):
    print("Addtion: ", t1 + t2, end="\n\n")
    print("Subtraction: ", t1 - t2, end="\n\n")
    return


@app.cell
def _(t1, t2):
    print("Multiplication:", t1 * t2, end="\n\n")
    print("Division:", t1 / t2, end="\n\n")
    return


@app.cell
def _(t1, t2, tf):
    print("Exponentiation:", tf.exp(t1), end="\n\n")
    print("Logarithm:", tf.math.log(t2), end="\n\n")
    return


if __name__ == "__main__":
    app.run()

