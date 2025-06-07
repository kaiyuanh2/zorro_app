# Zorro - Learning from Uncertain Data - From Possible Worlds to Possible Models

Zorro is a system for machine learning linear models over training and test data that is suffers from a variety of data quality issues. Given a training dataset with quality issues, e.g., missing values or outliers, Zorro treats the set of possible clean versions of the dataset as the possible worlds of an incomplete database and then trains all possible models and determines predication intervals for each test data points with respect to all possible models. As this is computationally infeasible, we instead use abstract interpretation to compactly over-approximate the space of possible training datasets, models, and inference results. Zorro utilizes Zonotopes, a type of convex polytope, to compactly represent these dataset variations, enabling the symbolic execution of gradient descent on all possible worlds simultaneously.

Users interact with the system as follows:

1. **Uploading training and test data**: the user uploads a training and test dataset and informs the system about the data quality issues in the dataset. For example, feature `age` has missing values. Zorro then bounds the possible clean versions of the dataset automatically using zonotopes.
2. **Model training and exploration**: Zorro then computes an abstract model where the model weights are zonotopes, guaranteeing that the optimal model for each possible training dataset are encompassed by the abstract model. In the Zorro's UI, the user can visually explore this abstract model, e.g., to investigate the range of possible weights for a feature through 2D and 3D projections of the zonotope polytopes or by directly inspecting the symbolic expressions of the abstract model.
3. **Inference and robustness evaluation**: Zorro then computes all possible predictions based on the possible models and shows them as predication intervals for each test data point. The system guarantees that for each test data point all possible outcomes are included in the predication intervals. The user can browse through the test prediction or analyze the robustness ratio of the model (given a user-specified tolerance on the width of prediction intervals, which fraction of test datapoints receive a robust prediction).

Zorro is the first system the empowers users to evaluate the robustness of their models when training data (and potentially also test data) is subject to data quality issues. That is, how robust is the model over all possible repairs of a training / test dataset.

# Demo

This demo is a node application that executes python code from within javascript using a library and uses libginac for symbolic computations. The system was tested on Ubuntu 22.04 with python 3.11. The system is also available through docker:

```sh
docker run --rm -p 3000:3000 iitdbgroup/zorro:latest
```

Then open `127.0.0.1:3000` in your browser.

## setup - install dependencies

### libginac

- Zorro uses libginac for symbolic computations

```sh
sudo apt-get install libginac11
```

### python

- create a virtual environment with python3.11

```sh
python -m venv venv
source ./venv/bin/activate
```

- install dependencies

```sh
pip install -r requirement.txt
```

### node

```sh
npm install
```

## run the demo

```sh
zorro.sh
```

Then open `127.0.0.1:3000` in your browser.

# References

- [Learning from Uncertain Data: From Possible Worlds to Possible Models. Jiongli Zhu, Su Feng, Boris Glavic and Babak Salimi. NeurIPS 2024.](http://www.cs.uic.edu/%7ebglavic/dbgroup/assets/pdfpubls/ZF24.pdf)
- [Zorro: Quantifying Uncertainty in Models & Predictions Arising from Dirty Data.
Kaiyuan Hu, Jiongli Zhu, Boris Glavic and Babak Salimi. SIGMOD 2025 demo](http://www.cs.uic.edu/%7ebglavic/dbgroup/assets/pdfpubls/HZ25.pdf)
