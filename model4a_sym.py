import sympy as sym
from utils import Utils

class Model4aSymbolic:
    def __init__(self):
        self.u = sym.symbols(f"u1:81")
        self.g = sym.symbols(f"g1:81")
        self.a = sym.symbols(f"a1:151")

        self.objective_function = self.function()
        self.gradient = Utils().gradient(self.objective_function, self.u)

        self.objective_function_file = "model4a_objective_function.pkl"
        self.gradient_file = "model4a_gradient.pkl"

    def function(self):
        K = lambda x: 2 * ((x - 1) // 9)
        D = lambda x, y : (x**2 + y**2)**0.5

        u, g, a = self.u, self.g, self.a

        return -sym.Matrix(g).dot(sym.Matrix(u)) + sum(a[i-1] * (D(u[2*i-2], 1 + u[2*i-1]) - 1)**2 for i in range(1, 11)) \
            + sum(a[i+9] * (D(1 + u[2*i], 1 + u[2*i+1]) - 2**0.5)**2 for i in range(1, 10)) \
            + sum(a[i+18] * (D(1 - u[2*i-2], 1 + u[2*i-1]) - 2**0.5)**2 for i in range(1, 10)) \
            + sum(a[i+27] * (D(1 + u[2*i+19] - u[2*i-1],  u[2*i+18] - u[2*i-2]) - 1)**2 for i in range(1, 31)) \
            + sum(a[i+57] * (D(1 + u[2*i+K(i)] - u[2*i-2+K(i)], u[2*i+1+K(i)] - u[2*i-1+K(i)]) - 1)**2 for i in range(1, 37)) \
            + sum(a[i+93] * (D(1 + u[2*i+20+K(i)] - u[2*i-2+K(i)], 1 + u[2*i+21+K(i)] - u[2*i-1+K(i)]) - 2**0.5)**2 for i in range(1, 28)) \
            + sum(a[i+120] * (D(1 - u[2*i+18+K(i)] + u[2*i+K(i)], 1 + u[2*i+19+K(i)] - u[2*i+1+K(i)]) - 2**0.5)**2 for i in range(1, 28))

def main():
    model = Model4aSymbolic()
    utils = Utils()

    m = sym.lambdify((model.u, model.g, model.a), model.objective_function, 'numpy')
    f = sym.lambdify((model.u, model.g, model.a), model.gradient, 'numpy')
    
    utils.serialize(m, model.objective_function_file)
    utils.serialize(f, model.gradient_file)

    print(f"Dumped lambdified objective function and gradient to '{model.objective_function_file}' and '{model.gradient_file}'")

if __name__ == "__main__":  
    main()