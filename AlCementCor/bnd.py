import fenics as fe
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Tuple, List

from AlCementCor.material_model_config import LinearElastoPlasticConfig


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


class ConstantStrainRateExp(fe.UserExpression):
    def __init__(self, strain_rate, **kwargs):
        super().__init__(**kwargs)
        self.strain_rate = strain_rate
        self.time = 0.0

    def eval(self, values, x):
        values[0] = self.strain_rate * self.time

    def update_time(self, time_step):
        self.time = time_step


# Define class for Constant Strain Rate boundary condition
class ConstantStrainRateBCX(BoundaryCondition):
    def __init__(self, V, on_boundary, strain_rate):
        super().__init__(V, on_boundary)
        self.strain_rate_expr = ConstantStrainRateExp(strain_rate, degree=0)

    def get_condition(self):
        return fe.DirichletBC(self.V.sub(0), self.strain_rate_expr, self.on_boundary)

    def get_homogenized_condition(self):
        return fe.DirichletBC(self.V.sub(0), 0, self.on_boundary)

    def update_time(self, time_step):
        self.strain_rate_expr.update_time(time_step)


# Define class for Constant Strain Rate boundary condition
class ConstantStrainRateBCY(BoundaryCondition):
    def __init__(self, V, on_boundary, strain_rate):
        super().__init__(V, on_boundary)
        self.strain_rate_expr = ConstantStrainRateExp(strain_rate, degree=0)

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


class ConstantStrainRate(fe.UserExpression):
    def __init__(self, strain_rate, **kwargs):
        super().__init__(**kwargs)
        self.strain_rateX = strain_rate[0]
        self.strain_rateY = strain_rate[1]
        self.time = 1e-16

    def eval(self, values, x):
        # linearly increasing displacement in the x-direction with respect to time and x-coordinate
        values[0] = self.time * self.strain_rateX
        values[1] = self.time * self.strain_rateY

    def update_time(self, time_step):
        self.time = time_step

    def value_shape(self):
        return (2,)


class SquareStrainRate(fe.UserExpression):
    def __init__(self, strain_rate, start, end, minimum_displacement=0.0, **kwargs):
        super().__init__(**kwargs)
        self.strain_rateX = strain_rate[0]
        self.strain_rateY = strain_rate[1]
        self.start = start
        self.end = end
        self.time = 1e-16
        self.min_disp = minimum_displacement

    def eval(self, values, x):
        # linearly increasing displacement in the x-direction with respect to time and x-coordinate
        values[0] = self.time * self.strain_rateX * ((1.0 - self.min_disp) * (
                    1.0 - (2.0 / (self.end - self.start) * (x[1] - self.start) - 1) ** 2) + self.min_disp)
        values[1] = self.time * self.strain_rateY * (
                    1.0 - (2.0 / (self.end - self.start) * (x[0] - self.start) - 1) ** 2)

    def update_time(self, time_step):
        self.time = time_step

    def value_shape(self):
        return (2,)


class LinearDisplacementX(fe.UserExpression):
    def __init__(self, strain_rate, bnd_length, **kwargs):
        super().__init__(**kwargs)
        self.strain_rateX = strain_rate[0]
        self.strain_rateY = strain_rate[1]
        self.bnd_length = bnd_length
        self.time = 1e-16

    def eval(self, values, x):
        # linearly increasing displacement in the x-direction with respect to time and x-coordinate
        values[0] = self.time * (self.strain_rateY * x[1] + self.strain_rateX * x[0]) / self.bnd_length
        values[1] = 0.0

    def update_time(self, time_step):
        self.time = time_step

    def value_shape(self):
        return (2,)


class LinearDisplacementY(fe.UserExpression):
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


class SinglePointDisplacement(fe.UserExpression):
    def __init__(self, point, strain_rate, **kwargs):
        super().__init__(**kwargs)
        self.strain_rateX = strain_rate[0]
        self.strain_rateY = strain_rate[1]
        self.point = point
        self.time = 1e-16

    def eval(self, values, x):
        distance = np.linalg.norm(self.point - x)
        if distance < 1.0:
            values[0] = self.time * self.strain_rateX
            values[1] = self.time * self.strain_rateY
        elif self.time < 10.0:
            values[0] = self.time * self.strain_rateX * ((6.0 - distance) / 6.0) ** 7
            values[1] = self.time * self.strain_rateY * ((6.0 - distance) / 6.0) ** 7

    def update_time(self, time_step):
        self.time = time_step

    def value_shape(self):
        return (2,)


class TimeDependentStress(fe.UserExpression):
    def __init__(self, initial_value, **kwargs):
        super().__init__(**kwargs)
        self.stress_value = initial_value

    def eval(self, value, x):
        value[0] = self.stress_value[0]
        value[1] = self.stress_value[1]

    def update(self, new_value):
        self.stress_value = new_value

    def value_shape(self):
        return (2,)


class BaseElastoPlasticBnd:
    def __init__(self, simulation_config: "LinearElastoPlasticConfig", V: Any) -> None:
        self.simulation_config = simulation_config
        self.mesh = simulation_config.mesh
        self.V = V
        self.l_x = simulation_config.l_x
        self.l_y = simulation_config.l_y

    def make_top_boundary(self) -> Any:
        """Defines the top boundary condition."""
        l_y = self.l_y

        def is_top_boundary(x: Tuple[float, float], on_boundary: bool) -> bool:
            return on_boundary and fe.near(x[1], l_y)

        return is_top_boundary

    def make_bottom_boundary(self) -> Any:
        def is_bottom_boundary(x: Tuple[float, float], on_boundary: bool) -> bool:
            return on_boundary and fe.near(x[1], 0.0)

        return is_bottom_boundary

    def make_left_boundary(self) -> Any:
        def is_left_boundary(x: Tuple[float, float], on_boundary: bool) -> bool:
            return on_boundary and fe.near(x[0], 0.0)

        return is_left_boundary

    def make_right_boundary(self) -> Any:
        l_x = self.l_x  # Storing value to be used in closure

        def is_right_boundary(x: Tuple[float, float], on_boundary: bool) -> bool:
            return on_boundary and fe.near(x[0], l_x)

        return is_right_boundary

    @abstractmethod
    def update_bnd(self, time_step, time):
        pass


class DisplacementElastoPlasticBnd(BaseElastoPlasticBnd):
    def __init__(self, simulation_config, V):
        """Initializes the displacement elasto-plastic boundary conditions."""
        super().__init__(simulation_config, V)
        self.bc, self.bc_iter, self.conditions = self.setup_displacement_bnd()

    def setup_displacement_bnd(self) -> Tuple[List[Any], List[Any], List[Any]]:
        # Define the boundary conditions
        # bnd_length = l_x
        # displacement_func = LinearDisplacementX((-C_strain_rate, 0.0), bnd_length)
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        # bottom_condition = FunctionDisplacementBoundaryCondition(V, is_bottom_boundary, displacement_func)
        # bottom_condition = NoDisplacementBoundaryCondition(V, is_bottom_boundary)
        # if two_layers:
        #     bottom_condition = ConstantStrainRateBoundaryCondition(V, is_bottom_boundary, -C_strain_rate)
        # else:
        #     bottom_condition = NoDisplacementBoundaryCondition(V, is_bottom_boundary)

        # top_condition = ConstantStrainRateBoundaryCondition(V, is_top_boundary, C_strain_rate)
        # bnd_length = 100.0
        # displacement_func = LinearDisplacementX(-C_strain_rate, bnd_length)
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        # top_condition = FunctionDisplacementBoundaryCondition(V, is_top_boundary, displacement_func)
        top_condition = NoDisplacementBoundaryCondition(self.V, self.make_top_boundary())

        # bnd_length = l_y
        # displacement_func = SinglePointDisplacement((0.0, 6.0), (-C_strain_rate, 0.0))
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        # displacement_func = SquareStrainRate((-self.simulation_config.strain_rate, 0.0), 0.0, self.l_y)
        # left_condition = FunctionDisplacementBoundaryCondition(self.V, is_left_boundary, displacement_func)

        # displacement_func = SinglePointDisplacement((4.2, 6.0), (-C_strain_rate, 0.0))
        # displacement_func = ConstantStrainRate((-C_strain_rate, 0.0))
        # right_condition = FunctionDisplacementBoundaryCondition(V, is_right_boundary, displacement_func)

        # Create the conditions list
        conditions = [top_condition]

        # Generate the Dirichlet boundary conditions
        bc = [condition.get_condition() for condition in conditions]

        # Generate homogenized boundary conditions
        bc_iter = [condition.get_homogenized_condition() for condition in conditions]

        return bc, bc_iter, conditions

    def update_bnd(self, time_step, time):
        if self.conditions is None:
            return

        for condition in self.conditions:
            condition.update_time(time_step)


class StressElastoPlasticBnd(BaseElastoPlasticBnd):
    def __init__(self, simulation_config, V, initial_stress_value=None):
        super().__init__(simulation_config, V)
        if initial_stress_value is None:
            initial_stress_value = [0.0, 0.1]
        self.boundary_markers = None
        self.stress_expression = TimeDependentStress(initial_stress_value, degree=0)
        self.setup_stress_bnd()

    def setup_stress_bnd(self) -> None:
        self.boundary_markers = fe.MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        self.boundary_markers.set_all(0)

        # For more complex boundary conditions, consider using 'fe.AutoSubDomain' with Python functions.
        # top_boundary = fe.AutoSubDomain(lambda x, on_boundary: is_top_boundary(x, on_boundary, self.l_y))
        bottom_boundary = fe.AutoSubDomain(self.make_bottom_boundary())

        # boundary_subdomain = fe.CompiledSubDomain("near(x[1], 0.0)")
        bottom_boundary.mark(self.boundary_markers, 1)

    def update_stress_value(self, new_stress_value, time) -> None:
        """Updates the stress value.

        Args:
            new_stress_value (list): New stress values to be set.
        """
        self.stress_expression.update(new_stress_value)

    def get_stress_rhs(self, test_function):
        ds = fe.Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        return fe.dot(test_function, self.stress_expression) * ds(1)

    def update_bnd(self, time_step, time):
        self.update_stress_value([0.0, time * 0.1], time)

