import numpy as np
from skl2onnx import to_onnx
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from smartredis import Client


def build_lin_reg():
    x = np.array([[1.0], [2.0], [6.0], [4.0], [3.0], [5.0]]).astype(np.float32)
    y = np.array([[2.0], [3.0], [7.0], [5.0], [4.0], [6.0]]).astype(np.float32)

    linreg = LinearRegression()
    linreg.fit(x, y)
    linreg = to_onnx(linreg, x.astype(np.float32), target_opset=13)
    return linreg.SerializeToString()


def build_kmeans():

    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    tr = KMeans(n_clusters=2)
    tr.fit(X)

    kmeans = to_onnx(tr, X, target_opset=11)
    return kmeans.SerializeToString()


def build_random_forest():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=13)
    clr = RandomForestRegressor(n_jobs=1, n_estimators=100)
    clr.fit(X_train, y_train)

    rf_model = to_onnx(clr, X_test.astype(np.float32), target_opset=13)
    return rf_model.SerializeToString()


def run_model(client: Client, device, model, model_input, in_name, out_names):
    client.put_tensor(in_name, model_input)
    client.set_model("onnx_model", model, "ONNX", device=device)
    client.run_model("onnx_model", inputs=[in_name], outputs=out_names)
    outputs = []
    for o in out_names:
        outputs.append(client.get_tensor(o))
    return outputs


def run(device):

    # connect a client to the database
    client = Client(cluster=False)

    # linreg test
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]).astype(np.float32)
    linreg = build_lin_reg()
    outputs = run_model(client, device, linreg, X, "X", ["Y"])
    assert len(outputs[0]) == 5
    print("Linear regression successful")
    print(outputs)

    # Kmeans test
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    kmeans = build_kmeans()
    outputs = run_model(
        client, device, kmeans, X, "kmeans_in", ["kmeans_labels", "kmeans_transform"]
    )
    assert len(outputs) == 2
    assert len(outputs[0]) == 10
    assert outputs[1].shape == (10, 2)
    print("K-means successful")
    print(outputs)

    # test random Forest regressor
    sample = np.array([[6.4, 2.8, 5.6, 2.2]]).astype(np.float32)
    model = build_random_forest()
    outputs = run_model(client, device, model, sample, "rf_in", ["rf_label"])
    assert len(outputs[0]) == 1
    print("Random forest successful")
    print(outputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX test Script")
    parser.add_argument(
        "--device", type=str, default="CPU", help="device type for model execution"
    )
    args = parser.parse_args()
    run(args.device)
