import numpy as np
from rosenbrock_function_sym import RosenbrockFunctionSymbolic
from model4a_sym import Model4aSymbolic
from utils import Utils

class Homework1:
    def __init__(self):
        self.u1 = np.array([-0.75, 0.7])

        self.u2 = np.array([
            0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649,
            0.1576, 0.9706, 0.9572, 0.4854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595,
            0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
            0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344,
            0.4387, 0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547,
            0.2760, 0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238,
            0.7513, 0.2551, 0.5060, 0.6991, 0.8909, 0.9593, 0.5472, 0.1386, 0.1493, 0.2575,
            0.8407, 0.2543, 0.8143, 0.2435, 0.9293, 0.3500, 0.1966, 0.2511, 0.6160, 0.4733
        ])
        self.g2 = np.array([0.0]*80)
        self.g2[61] = 1.0
        self.g2[78] = 1.0
        self.a2 = np.array([1.0]*150)
    
    def test(self):
        utils = Utils()
        model1 = RosenbrockFunctionSymbolic()
        model2 = Model4aSymbolic()

        rosenbrock_objective = utils.deserialize(model1.objective_function_file)
        rosenbrock_gradient = utils.deserialize(model1.gradient_file)

        model4a_objective = utils.deserialize(model2.objective_function_file)
        model4a_gradient = utils.deserialize(model2.gradient_file)

        m1_expected = 3.2516
        f1_expected = np.array([0.6250, 2.7500])

        m1 = utils.calculate(rosenbrock_objective, self.u1)
        f1 = utils.calculate(rosenbrock_gradient, self.u1)

        m2_expected = 37.9826
        f2_expected = np.array([
            0.8276,  3.9456,  -4.9622, 4.8403,  2.9894,  -2.6788, -1.2564, 2.0678,  5.9835,  6.4761,
            -2.9499, 5.9816,  4.0644,  2.5033,  2.7696,  -0.9908, 1.1392,  5.1424,  4.2488,  4.7884,
            1.6421,  -3.9092, 1.8731,  3.6070,  0.3175,  2.2978,  1.5792,  -1.3590, 3.0317,  -3.5712,
            -0.2933, -2.7038, -1.9176, -1.6771, -4.4043, 3.6004,  1.7816,  -1.7667, -0.2797, -1.6397,
            -1.2903, 0.2238,  1.4346,  0.5476,  -3.5096, -1.6091, -0.1381, 2.5677,  0.7389,  3.1101,
            -2.8609, 3.3196,  2.5639,  -0.9866, -3.5648, 1.1061,  3.5476,  0.3216,  -0.5678, -0.4766,
            0.7133,  -1.2582, -0.6056, -0.0006, 1.5608,  1.6810,  1.1494,  -0.9432, -2.2373, -1.2449,
            0.9007,  -0.5076, 0.6174,  -0.8126, 2.2608,  0.2671,  -3.1128, 0.9078,  -0.2301, 0.4531
        ])

        m2 = utils.calculate(model4a_objective, self.u2, self.g2, self.a2)
        f2 = utils.calculate(model4a_gradient, self.u2, self.g2, self.a2)

        print(f"Close to expected?\n"
                "-- Rosenbrock Function: --\n"
                f"Objective function: {np.allclose(m1, m1_expected, rtol=0, atol=1e-4)}\n"
                f"Gradient: {np.allclose(f1, f1_expected, rtol=0, atol=1e-4)}\n"
                "-- Model 4a: --\n"
                f"Objective function: {np.allclose(m2, m2_expected, rtol=0, atol=1e-4)}\n"
                f"Gradient: {np.allclose(f2, f2_expected, rtol=0, atol=1e-4)}\n")

def main():
    homework = Homework1()
    homework.test()

if __name__ == "__main__":
    main()
