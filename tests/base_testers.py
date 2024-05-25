"""
created matt_dumont 
on: 24/05/24
"""
import numpy as np

class BaseTesterCounter:
    @classmethod
    def make_step_test_data(self, delta, length):
        x1 = np.full(length, 12)
        x2 = x1 + delta
        return x1, x2

    @classmethod
    def make_linear_test_data(self, slope, length, delta=0):
        x1 = np.full(length, 12)
        x = np.arange(length)
        x2 = x1 + slope * x + delta
        return x1, x2

    @classmethod
    def make_bilinar_test_data(self, slope1, slope2, length, delta1=0, delta2=0):
        x = np.arange(length)
        x0 = np.full(length, 12)
        x1 = x0 + slope1 * x + delta1
        x2 = x0 + slope2 * x + delta2
        return x1, x2

