import numpy as np
from rosenbrock_function_sym import RosenbrockFunctionSymbolic
from model4a_sym import Model4aSymbolic
from utils import Utils

class Homework2:
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
        rosenbrock = RosenbrockFunctionSymbolic()
        model4a = Model4aSymbolic()

        rosenbrock_objective = utils.deserialize(rosenbrock.objective_function_file)
        rosenbrock_gradient = utils.deserialize(rosenbrock.gradient_file)

        model4a_objective = utils.deserialize(model4a.objective_function_file)
        model4a_gradient = utils.deserialize(model4a.gradient_file)

        m1_expected = 3.57e-29
        u1_expected = np.array([1.0, 1.0])

        m2_expected = -0.9780
        u2_expected = np.array([
            0.1148, 0.1673,  0.0489, 0.1065,  0.0329, 0.0685,  0.0436, 0.0390,
            0.0639, 0.0226,  0.0815, 0.0161,  0.0904, 0.0102,  0.0876, -0.0087,
            0.0851, -0.0522, 0.1221, -0.1484, 0.2031, 0.3337,  0.1275, 0.2305,
            0.1032, 0.1254,  0.1197, 0.0553,  0.1459, 0.0254,  0.1750, 0.0166,
            0.2054, 0.0099,  0.2329, -0.0173, 0.2451, -0.1062, 0.2612, -0.2662,
            0.3275, 0.5355,  0.2346, 0.3367,  0.2199, 0.1473,  0.2196, 0.0561,
            0.2336, 0.0171,  0.2653, 0.0020,  0.3189, -0.0125, 0.3968, -0.0511,
            0.4832, -0.1454, 0.4773, -0.3626, 0.5957, 0.8310,  0.4409, 0.3355,
            0.3709, 0.1572,  0.3208, 0.0591,  0.3065, 0.0093,  0.3326, -0.0211,
            0.4031, -0.0557, 0.5233, -0.1183, 0.7062, -0.2429, 0.9915, -0.5251
        ])

        m1_vanilla, u1_vanilla = utils.gradient_descent(rosenbrock_objective, rosenbrock_gradient, self.u1, round=4, alpha=0.1)
        m1_armijo, u1_armijo = utils.gradient_descent_armijo(rosenbrock_objective, rosenbrock_gradient, self.u1, round=4, alpha_init=1, r=0.5, c=0.5)

        m2_vanilla, u2_vanilla = utils.gradient_descent(model4a_objective, model4a_gradient, self.u2, self.g2, self.a2, round=4, alpha=0.1)
        m2_armijo, u2_armijo = utils.gradient_descent_armijo(model4a_objective, model4a_gradient, self.u2, self.g2, self.a2, round=4, alpha_init=1, r=0.5, c=0.5)

        print(f"Close to expected?\n"
                "-- Rosenbrock Function: --\n"
                f"Objective function (no optimization): {np.allclose(m1_vanilla, m1_expected)}\n"
                f"Parameters (no optimization): {np.allclose(u1_vanilla, u1_expected)}\n"
                f"Objective function (Armijo): {np.allclose(m1_armijo, m1_expected)}\n"
                f"Parameters (Armijo): {np.allclose(u1_armijo, u1_expected)}\n"
                "-- Model 4a: --\n"
                f"Objective function (no optimization): {np.allclose(m2_vanilla, m2_expected, rtol=0, atol=1e-4)}\n"
                f"Parameters (no optimization): {np.allclose(u2_vanilla, u2_expected, rtol=0, atol=1e-4)}\n"
                f"Objective function (Armijo): {np.allclose(m2_armijo, m2_expected, rtol=0, atol=1e-4)}\n"
                f"Parameters (Armijo): {np.allclose(u2_armijo, u2_expected, rtol=0, atol=1e-4)}\n")

def main():
    homework = Homework2()
    homework.test()

if __name__ == "__main__":
    main()
