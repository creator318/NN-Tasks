import marimo

__generated_with = "0.14.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return (
        LinearRegression,
        fetch_california_housing,
        math,
        mean_absolute_error,
        mean_squared_error,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
    )


@app.cell
def _(fetch_california_housing, pd):
    housing = fetch_california_housing()
    _X = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    _y = pd.DataFrame(data=housing.target, columns=housing.target_names)
    data = pd.concat([_X, _y], axis=1)
    return (data,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data, pd):
    data.dropna(inplace=True)
    if 'ocean_proximity' in data.columns:
        data_1 = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
    return (data_1,)


@app.cell
def _(data_1):
    data_1.describe()
    return


@app.cell
def _(data_1, plt, sns):
    sns.heatmap(data_1.corr(), annot=True, cmap=plt.cm.coolwarm, fmt='.2f')
    plt.show()
    return


@app.cell
def _(data_1, plt):
    data_1.hist(figsize=(12, 8), bins=50)
    plt.show()
    return


@app.cell
def _(data_1, math, plt, sns):
    boxdata = data_1.drop(columns=data_1.filter(like='ocean_proximity').columns)
    nFeatures = len(boxdata.columns)
    rows = math.ceil(nFeatures / 3)
    cols = min(3, nFeatures)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes]
    for i, col in enumerate(boxdata.columns):
        sns.boxplot(boxdata[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel(col)
    plt.show()
    return


@app.cell
def _(data_1, train_test_split):
    if 'X' not in locals() or 'y' not in locals():
        _X = data_1.drop(columns=['median_house_value'])
        _y = data_1['median_house_value']
    XTrain, XTest, yTrain, yTest = train_test_split(_X, _y, test_size=0.2, random_state=0)
    return XTest, XTrain, yTest, yTrain


@app.cell
def _(LinearRegression, XTrain, yTrain):
    model = LinearRegression()
    model.fit(XTrain, yTrain)

    print("Model Params:")
    print(" Weights:", model.coef_)
    print(" Bias:", model.intercept_)
    return (model,)


@app.cell
def _(XTest, mean_absolute_error, mean_squared_error, model, r2_score, yTest):
    yPred = model.predict(XTest)

    mae = mean_absolute_error(yTest, yPred)
    mse = mean_squared_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    print("Model Metrics:")
    print(f" Mean Absolute Error (MAE): {mae}")
    print(f" Mean Squared Error (MSE): {mse}")
    print(f" R-Squared Score: {r2}")
    return (yPred,)


@app.cell
def _(plt, yPred, yTest):
    plt.scatter(x=yTest, y=yPred, alpha=0.3, s=20, linewidth=0)
    plt.xlabel('Actual House Value')
    plt.ylabel('Predicted House Value')
    plt.title('Actual vs. Predicted House Prices')
    plt.show()
    return


if __name__ == "__main__":
    app.run()

