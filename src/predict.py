"""Medical Insurance Cost Prediction app - Day 22, Phase 7.
Uses the trained Multiple Linear Regression model (Day 20, theta from
linear_regression_model.npz) to predict insurance charges."""
import numpy as np

_theta = np.load("linear_regression_model.npz")["theta"] if False else None
import os
_here = os.path.dirname(os.path.abspath(__file__))
# theta saved from Day 20 training (intercept + 9 coefficients)
THETA = np.array([-9251.20197709, 248.22007025, -101.91470016, 318.6730785, 532.85657593, 23077.55053583, -1839.76547773, -2232.33931234, -2679.38496726, -2499.71221977])

def predict(age, sex, bmi, children, smoker, region):
    reg = {'northeast': [1,0,0,0], 'northwest': [0,1,0,0],
           'southeast': [0,0,1,0], 'southwest': [0,0,0,1]}[region.lower()]
    x = np.array([age, 1 if sex.lower() == 'male' else 0, bmi, children,
                  1 if smoker.lower() == 'yes' else 0] + reg, dtype=float)
    return float(np.r_[1.0, x] @ THETA)

if __name__ == "__main__":
    cases = [
        (35, 'male', 28.0, 2, 'no', 'southeast'),
        (45, 'female', 32.0, 1, 'yes', 'northwest'),
        (25, 'female', 22.0, 0, 'no', 'southwest'),
        (55, 'male', 30.5, 3, 'yes', 'southeast'),
    ]
    for i, c in enumerate(cases, 1):
        print(f"Case {i}: age {c[0]}, {c[1]}, BMI {c[2]}, children {c[3]}, "
              f"smoker {c[4]}, {c[5]} -> predicted charges ${predict(*c):,.2f}")
