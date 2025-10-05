import sympy as sym
from utils import Utils

class RosenbrockFunctionSymbolic:
    def __init__(self):
        self.u = sym.symbols(f"u1:3")
    
        self.objective_function = self.function()
        self.gradient = Utils().gradient(self.objective_function, self.u)

        self.objective_function_file = "rosenbrock_objective_function.pkl"
        self.gradient_file = "rosenbrock_gradient.pkl"

    def function(self):
        u = self.u
        return 10*(u[1] - u[0]**2)**2 + (u[0] - 1)**2

def main():
    model = RosenbrockFunctionSymbolic()
    utils = Utils()

    m = sym.lambdify(model.u, model.objective_function, 'numpy')
    f = sym.lambdify(model.u, model.gradient, 'numpy')

    utils.serialize(m, model.objective_function_file)
    utils.serialize(f, model.gradient_file)

    print(f"Dumped lambdified objective function and gradient to '{model.objective_function_file}' and '{model.gradient_file}'")

if __name__ == "__main__":  
    main()