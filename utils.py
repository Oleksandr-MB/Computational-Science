import numpy as np
import sympy as sym
import dill

class Utils:
    def __init__(self):
        pass

    def serialize(self, obj, filename):
        with open(filename, "wb") as file:
            dill.dump(obj, file)

    def deserialize(self, filename):
        with open(filename, "rb") as file:
            return dill.load(file)
        
    def calculate(self, f, *args):
        return f(*args[0]) if len(args) == 1 else f(*args)


    def gradient(self, f, *args):
        return sym.derive_by_array(f, *args)

    def gradient_descent(self, m, f, *args, alpha=0.1):
        u, consts = args[0].copy(), args[1:]
        
        m_new = self.calculate(m, *args)
        m_old = float('inf')

        while m_new < m_old:
            m_old = m_new
            f_curr = self.calculate(f, u, *consts).flatten()
            h = -f_curr
            u = u + alpha * h

            m_new = self.calculate(m, u, *consts)
        
        u = u - alpha * h
  
        return m_old, u
    
    def gradient_descent_armijo(self, m, f, *args, alpha_init=1, r=0.5, c=0.5):
        u, consts = args[0].copy(), args[1:]
        m_new = self.calculate(m, *args)
        m_old = float('inf')
        alpha = alpha_init
        f_curr = np.zeros_like(u)
        h = np.zeros_like(u)
        while m_new < m_old:
            m_old = m_new
            f_curr = self.calculate(f, u, *consts).flatten()
            h = -f_curr
            alpha = 1/r * alpha_init
            m_x = float('inf')

            while m_x >  m_new + c * alpha * h.dot(f_curr):
                alpha = alpha * r
                arg1_x = u + alpha * h
                m_x = self.calculate(m, arg1_x, *consts)

            m_new = m_x
            u = arg1_x

        u = u - alpha * h
        
        return m_old, u

    def conjugate_gradient_descent(self, m, f, *args, beta_n=1, alpha_init=1, r=0.5, c=0.5, n_reset=20):
        epsilon = 1e-12 # To avoid division by zero
        beta1 = lambda f_new, f_old: f_new.dot(f_new) / max(f_old.dot(f_old), epsilon) # 4.6
        beta2 = lambda f_new, f_old: f_new.dot(f_new - f_old) / max(f_old.dot(f_old), epsilon) # 4.7
        beta3 = lambda f_new, f_old, h_old: f_new.dot(f_new - f_old) / max(h_old.dot(f_new - f_old), epsilon) # 4.8
        beta4 = lambda f_new, f_old, h_old: f_new.dot(f_new) / max(h_old.dot(f_new - f_old), epsilon) # 4.9

        u, consts = args[0].copy(), args[1:]

        count = 0
        m_new = self.calculate(m, *args)
        m_old = float('inf')
        alpha = alpha_init
        f_new = np.zeros_like(u)
        h_new = np.zeros_like(u)

        while m_new < m_old:
            m_old = m_new
            f_old = f_new
            h_old = h_new
            f_new = self.calculate(f, u, *consts).flatten()

            if count % n_reset == 0:
                h_new = -f_new
            else:
                if beta_n == 1:
                    beta = beta1(f_new, f_old)
                elif beta_n == 2:
                    beta = beta2(f_new, f_old)
                elif beta_n == 3:
                    beta = beta3(f_new, f_old, h_old)
                elif beta_n == 4:
                    beta = beta4(f_new, f_old, h_old)
                else:
                    raise ValueError("Invalid beta_n value. Choose from {1, 2, 3, 4}.")

                h_new = -f_new + max(0, beta) * h_old
            
            alpha = 1/r * alpha_init
            m_x = float('inf')

            while m_x >  m_new + c * alpha * h_new.dot(f_new):
                alpha = alpha * r
                u_x = u + alpha * h_new
                m_x = self.calculate(m, u_x, *consts)

            m_new = m_x
            u = u_x
            count += 1

        u = u - alpha * h_new
        
        return m_old, u