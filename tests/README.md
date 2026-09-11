# Regression tests

Run from the NFL project root:

    python -m unittest discover -s tests -t .

Run one module:

    python -m unittest tests.test_joint_feature_selection

Some tests fit small TensorFlow models. Avoid running the full suite alongside
a large backtest; discovery alone does not execute tests.
