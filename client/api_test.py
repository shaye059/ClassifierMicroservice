import requests
import unittest

import numpy


class ApiTestCase(unittest.TestCase):
    HOST = "http://api:8888"

    def test_0_health(self):
        """Sanity check to ensure that the API is reachable"""
        resp = self._request("GET", "/health/")
        assert resp["status"] == "ok"

    def test_1_read_write(self):
        """test read/write operation"""
        # create
        resp = self._request(
            "POST",
            "/models/",
            {
                "model": "SGDClassifier",
                "params": {"alpha": 0.0001, "penalty": "l1"},
                "d": 4,
                "n_classes": 3,
            },
        )
        model_id = resp["id"]
        # read
        resp = self._request("GET", f"/models/{model_id}/")
        assert resp["model"] == "SGDClassifier"
        assert resp["params"]["penalty"] == "l1"
        assert resp["d"] == 4
        assert resp["n_classes"] == 3
        assert numpy.isclose(resp["params"]["alpha"], 0.0001)
        assert resp["n_trained"] == 0

    def test_2_wrong_write(self):
        """test erroneous write operation"""
        with self.assertRaises(Exception) as error:
            self._request(
                "POST",
                "/models/",
                {
                    "model": "ERROR",
                    "params": {"alpha": 0.0001},
                    "d": 4,
                    "n_classes": 2,
                },
            )
        assert error.exception.response.status_code == 400

    def test_3_wrong_read(self):
        """test erroneous read operation"""
        with self.assertRaises(Exception) as error:
            self._request("GET", "/models/123456789/")
        assert error.exception.response.status_code == 404

    def test_4_train_predict(self):
        """test train/predict operation"""
        n = 100
        d = 4
        # create
        resp = self._request(
            "POST",
            "/models/",
            {
                "model": "SGDClassifier",
                "params": {},
                "d": d,
                "n_classes": 2,
            },
        )
        model_id = resp["id"]
        # train
        X = numpy.random.randn(n, d)
        W = numpy.random.randn(d)
        Y = X.dot(W) > 0
        for i in range(n):
            self._request(
                "POST",
                f"/models/{model_id}/train/",
                {
                    "x": list(X[i, :]),
                    "y": int(Y[i]),
                },
            )
        # read
        resp = self._request("GET", f"/models/{model_id}/")
        assert resp["n_trained"] == n
        # predict
        xb64 = "WzEuMTEsMi4yMiwzLjMzLC00LjQ0XQ=="
        resp = self._request("GET", f"/models/{model_id}/predict/?x={xb64}")
        assert resp["y"] < 2

    def test_5_train_errors(self):
        """test erroneous train operation"""
        d = 4
        n_classes = 3
        # create
        resp = self._request(
            "POST",
            "/models/",
            {
                "model": "SGDClassifier",
                "params": {},
                "d": d,
                "n_classes": n_classes,
            },
        )
        model_id = resp["id"]
        # wrong x
        X = numpy.random.randn(d + 1)
        Y = numpy.random.randint(0, n_classes)
        with self.assertRaises(Exception) as error:
            self._request(
                "POST",
                f"/models/{model_id}/train/",
                {
                    "x": list(X),
                    "y": int(Y),
                },
            )
        assert error.exception.response.status_code == 400
        # wrong y
        X = numpy.random.randn(d)
        Y = n_classes
        with self.assertRaises(Exception) as error:
            self._request(
                "POST",
                f"/models/{model_id}/train/",
                {
                    "x": list(X),
                    "y": int(Y),
                },
            )
        assert error.exception.response.status_code == 400

    def test_6_train_score(self):
        """test train score"""
        d = 4
        models = [
            {"model": "MLPClassifier", "n_trained": 4, "training_score": 0.0},
            {"model": "CategoricalNB", "n_trained": 2, "training_score": 1.0},
            {"model": "MLPClassifier", "n_trained": 13, "training_score": 0.5},
            {"model": "MLPClassifier", "n_trained": 48, "training_score": 1.0},
        ]
        model_id_to_score = {}
        # create and train all models
        for model in models:
            # create
            resp = self._request(
                "POST",
                "/models/",
                {
                    "model": model["model"],
                    "params": {},
                    "d": d,
                    "n_classes": 2,
                },
            )
            model_id = resp["id"]
            model_id_to_score[model_id] = model["training_score"]
            # train
            X = numpy.abs(numpy.random.randn(model["n_trained"], d))
            W = numpy.random.randn(d)
            Y = 1 + X.dot(W) > 0
            for i in range(len(Y)):
                self._request(
                    "POST",
                    f"/models/{model_id}/train/",
                    {
                        "x": list(X[i, :]),
                        "y": int(Y[i]),
                    },
                )
        # get training_score
        resp = self._request("GET", f"/models/")
        resp_model_id_to_score = {
            model["id"]: model["training_score"] for model in resp["models"]
        }
        for m_id, training_score in model_id_to_score.items():
            resp_score = resp_model_id_to_score[m_id]
            assert numpy.isclose(training_score, resp_score)

    @classmethod
    def _request(self, method, path, json=None):
        url = f"{self.HOST}{path}"
        resp = requests.request(method, url, json=json, timeout=1)
        resp.raise_for_status()
        return resp.json()
