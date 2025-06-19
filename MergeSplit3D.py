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

    t1 = tf.reshape(tf.range(1, 25, dtype=tf.int32), (2, 3, 4))
    t2 = tf.reshape(tf.range(25, 49, dtype=tf.int32), (2, 3, 4))
    return t1, t2


@app.cell
def _(t1, t2):
    print("Tensor 1: ", t1, "\n")
    print("Tensor 2: ", t2, "\n")
    return


@app.cell
def _(t1, t2, tf):
    m1 = tf.concat([t1, t2], axis=0)
    print("Merge on axis 0:", m1)
    return (m1,)


@app.cell
def _(t1, t2, tf):
    m2 = tf.concat([t1, t2], axis=1)
    print("Merge on axis 1:", m2)
    return


@app.cell
def _(t1, t2, tf):
    m3 = tf.concat([t1, t2], axis=2)
    print("Merge on axis 2:", m3)
    return (m3,)


@app.cell
def _(m1, tf):
    s1 = tf.split(m1, 2, axis=0)
    for _i in range(2):
        print(f'Split tensor {_i}:', s1[_i], end='\n\n')
    return


@app.cell
def _(m3, tf):
    s2 = tf.split(m3, [1, 2, 5], axis=2)
    for _i in range(3):
        print(f'Split tensor {_i}:', s2[_i], end='\n\n')
    return


if __name__ == "__main__":
    app.run()

