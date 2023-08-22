import numpy as np
import ufl
import fenics as fe

from AlCementCor.config import SimulationConfig
from AlCementCor.input_file import process_abaqus_input_file, ExternalInput
from AlCementCor.interpolate import interpolate_displacements
from AlCementCor.material_properties import MaterialProperties
from AlCementCor.postproc import plot_strain_displacement, plot_displacement, plot_movement


class LinearElastoPlasticConfig:
    def __init__(self, config_file: str):
        # todo: move to config file
        self.strain_rate = fe.Constant(0.000001)
        self._simulation_config, self._substrate_properties, self._layer_properties = self.load_simulation_config(
            config_file)
        self._setup_geometry()

    def load_simulation_config(self, file_name):
        """Loads and initializes a SimulationConfig object from a JSON configuration file."""
        simulation_config = SimulationConfig(file_name)
        if simulation_config.field_input_file:
            input_file = process_abaqus_input_file(simulation_config.field_input_file, plot=True)
            simulation_config.width = input_file[ExternalInput.WIDTH.value]
            simulation_config.length = input_file[ExternalInput.LENGTH.value]

            center_yz_points_outside, center_yz_points_inside = self.determine_center_plane(input_file)

            coordinates_on_center_plane = []
            for outside_points_t, inside_points_t in zip(center_yz_points_outside, center_yz_points_inside):
                combined_points_t = outside_points_t + inside_points_t
                coordinates_on_center_plane.append(combined_points_t)

            displacement_x = input_file[ExternalInput.DISPLACEMENTX.value]
            displacement_y = input_file[ExternalInput.DISPLACEMENTY.value]
            displacement_z = input_file[ExternalInput.DISPLACEMENTZ.value]

            plot_strain_displacement(input_file)

            x_coordinates = input_file[ExternalInput.X.value]
            y_coordinates = input_file[ExternalInput.Y.value]
            z_coordinates = input_file[ExternalInput.Z.value]

            displacement_x_center, displacement_y_center = interpolate_displacements(
                coordinates_on_center_plane, x_coordinates, y_coordinates, z_coordinates, displacement_x,
                displacement_y,
                displacement_z)

            # Call the plot function
            plot_displacement(displacement_x_center, displacement_y_center)

            # Call the plot function
            plot_movement(coordinates_on_center_plane, displacement_x_center, displacement_y_center)

            # # Assuming time_values are extracted from your data
            # time_values = result[ExternalInput.TIME.value]
            #
            # # Create an InPlaneInterpolator instance for both in-plane displacements
            # in_plane_interpolator = InPlaneInterpolator(time_values, coordinates_on_center_plane, displacement_x_center,
            #                                             displacement_y_center)
            #
            # # Now you can query this interpolator for displacements at any time and point in the plane
            # query_time = 25.0  # for example
            # query_point = (5.0, 2.0, 3.0)  # for example, assuming it's within the bounds of your data
            # disp_at_query_time_and_point_1, disp_at_query_time_and_point_2 = in_plane_interpolator.get_displacement_at_point(
            #     query_time, query_point)

        substrate_properties = MaterialProperties('material_properties.json', simulation_config.material)
        if simulation_config.use_two_material_layers:
            layer_properties = MaterialProperties('material_properties.json', simulation_config.layer_material)
        return simulation_config, substrate_properties, layer_properties

    def determine_center_plane(self, result):
        x_coordinates = result[ExternalInput.X.value]
        y_coordinates = result[ExternalInput.Y.value]
        z_coordinates = result[ExternalInput.Z.value]

        # Extract the outside and inside point indices
        outside_indices = result[ExternalInput.OUTSIDE_P.value]
        inside_indices = result[ExternalInput.INSIDE_P.value]

        def get_center_yz_points(indices, x_coordinates, y_coordinates, z_coordinates, tolerance=1e-3):
            center_yz_points_per_timestep = []
            for t in range(x_coordinates.shape[1]):
                x_values_t = x_coordinates[indices, t]
                y_values_t = y_coordinates[indices, t]
                z_values_t = z_coordinates[indices, t]
                center_x_t = np.median(x_values_t)
                center_yz_points_t = [(x, y, z) for x, y, z in zip(x_values_t, y_values_t, z_values_t) if
                                      abs(x - center_x_t) < tolerance]
                if len(center_yz_points_t) != 3:
                    print("missing points")
                    exit(-1)
                center_yz_points_per_timestep.append(center_yz_points_t)
            return center_yz_points_per_timestep

        center_yz_points_outside = get_center_yz_points(outside_indices, np.array(x_coordinates),
                                                        np.array(y_coordinates), np.array(z_coordinates))
        center_yz_points_inside = get_center_yz_points(inside_indices, np.array(x_coordinates), np.array(y_coordinates),
                                                       np.array(z_coordinates))

        return center_yz_points_outside, center_yz_points_inside

    def _setup_geometry(self):
        """Sets up the geometry of the simulation, including layer thickness and mesh initialization."""
        l_layer_x = self._simulation_config.layer_1_thickness if self._simulation_config.use_two_material_layers else 0.0
        l_layer_y = 0.0
        self._l_x = self._simulation_config.width + l_layer_x
        # todo: hardcode
        # multiplier_y = 3.0
        # l_y = multiplier_y * simulation_config.length + l_layer_y
        self._l_y = self._simulation_config.length + l_layer_y
        self._mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(self.l_x, self.l_y),
                                      self._simulation_config.mesh_resolution_x,
                                      self._simulation_config.mesh_resolution_y)

    @property
    def mesh(self):
        return self._mesh

    @property
    def l_x(self):
        return self._l_x

    @property
    def l_y(self):
        return self._l_y

    @property
    def width(self):
        return self._simulation_config.width

    @property
    def finite_element_degree_u(self):
        return self._simulation_config.finite_element_degree_u

    @property
    def finite_element_degree_stress(self):
        return self._simulation_config.finite_element_degree_stress

    @property
    def substrate_properties(self):
        return self._substrate_properties

    @property
    def layer_properties(self):
        return self._layer_properties

    @property
    def substrate_yield_strength(self):
        return self._substrate_properties.yield_strength

    @property
    def layer_yield_strength(self):
        return self._layer_properties.yield_strength

    @property
    def substrate_shear_modulus(self):
        return self._substrate_properties.shear_modulus

    @property
    def layer_shear_modulus(self):
        return self._layer_properties.shear_modulus

    @property
    def substrate_linear_isotropic_hardening(self):
        return self._substrate_properties.linear_isotropic_hardening

    @property
    def layer_linear_isotropic_hardening(self):
        return self._layer_properties.linear_isotropic_hardening