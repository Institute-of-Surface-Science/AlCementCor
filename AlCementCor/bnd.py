import fenics as fe
from abc import ABC, abstractmethod


# Define base class for boundary conditions
class BoundaryCondition(ABC):
    def __init__(self, V, on_boundary):
        self.V = V
        self.on_boundary = on_boundary

    @abstractmethod
    def get_condition(self):
        pass

    @abstractmethod
    def get_homogenized_condition(self):
        pass

    @abstractmethod
    def update_time(self, time_step):
        pass


# Define class for No Displacement boundary condition
class NoDisplacementBoundaryCondition(BoundaryCondition):
    def get_condition(self):
        return fe.DirichletBC(self.V, fe.Constant((0.0, 0.0)), self.on_boundary)

    def get_homogenized_condition(self):
        return fe.DirichletBC(self.V.sub(1), 0, self.on_boundary)

    def update_time(self, time_step):
        pass


class StrainRateExpression(fe.UserExpression):
    def __init__(self, strain_rate, **kwargs):
        super().__init__(**kwargs)
        self.strain_rate = strain_rate
        self.time = 0.0

    def eval(self, values, x):
        values[0] = self.strain_rate * self.time

    def update_time(self, time_step):
        self.time = time_step


# Define class for Constant Strain Rate boundary condition
class ConstantStrainRateBoundaryCondition(BoundaryCondition):
    def __init__(self, V, on_boundary, strain_rate):
        super().__init__(V, on_boundary)
        self.strain_rate_expr = StrainRateExpression(strain_rate, degree=0)

    def get_condition(self):
        return fe.DirichletBC(self.V.sub(1), self.strain_rate_expr, self.on_boundary)

    def get_homogenized_condition(self):
        return fe.DirichletBC(self.V.sub(1), 0, self.on_boundary)

    def update_time(self, time_step):
        self.strain_rate_expr.update_time(time_step)


class FunctionDisplacementBoundaryCondition(BoundaryCondition):
    def __init__(self, V, on_boundary, displacement_function):
        super().__init__(V, on_boundary)
        self.displacement_function = displacement_function

    def get_condition(self):
        return fe.DirichletBC(self.V, self.displacement_function, self.on_boundary)

    def get_homogenized_condition(self):
        return fe.DirichletBC(self.V, fe.Constant((0.0, 0.0)), self.on_boundary)

    def update_time(self, time_step):
        self.displacement_function.update_time(time_step)


class DisplacementExpressionX(fe.UserExpression):
    def __init__(self, strain_rate, bnd_length, **kwargs):
        super().__init__(**kwargs)
        self.strain_rate = strain_rate
        self.bnd_length = bnd_length
        self.time = 1e-16

    def eval(self, values, x):
        # linearly increasing displacement in the x-direction with respect to time and x-coordinate
        values[0] = self.time * self.strain_rate * (self.bnd_length - x[0]) / self.bnd_length
        values[1] = 0.0

    def update_time(self, time_step):
        self.time = time_step

    def value_shape(self):
        return (2,)

class DisplacementExpressionY(fe.UserExpression):
    def __init__(self, strain_rate, bnd_length, **kwargs):
        super().__init__(**kwargs)
        self.strain_rate = strain_rate
        self.bnd_length = bnd_length
        self.time = 1e-16

    def eval(self, values, x):
        values[0] = 0.0
        # linearly increasing displacement in the y-direction with respect to time and y-coordinate
        values[1] = self.time * self.strain_rate * (self.bnd_length - x[0]) / self.bnd_length

    def update_time(self, time_step):
        self.time = time_step

    def value_shape(self):
        return (2,)