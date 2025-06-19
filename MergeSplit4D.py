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
    t = tf.reshape(tf.range(1, 361), (4, 6, 3, 5))
    t.shape
    return (t,)


@app.cell
def _(t, tf):
    t1 = tf.reshape(t, (4 * 6, 3, 5))
    t1.shape
    return (t1,)


@app.cell
def _(t, tf):
    t2 = tf.reshape(t, (4, 6, 3 * 5))
    t2.shape
    return (t2,)


@app.cell
def _(t, tf):
    t3 = tf.concat([t, t], axis=1)
    t3.shape
    return (t3,)


@app.cell
def _(t1, tf):
    _t1 = tf.reshape(t1, (4, 6, 3, 5))
    _t1.shape
    return


@app.cell
def _(t2, tf):
    _t2 = tf.split(t2, [4, 11], axis=2)
    _t2[0].shape, _t2[1].shape
    return


@app.cell
def _(t3, tf):
    _t3 = tf.split(t3, 2, axis=1)
    _t3[0].shape, _t3[1].shape
    return


if __name__ == "__main__":
    app.run()

