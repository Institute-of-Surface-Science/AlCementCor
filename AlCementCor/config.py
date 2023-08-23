import json
import os
from enum import Enum
from typing import Any, Dict


class Direction(Enum):
    X = [1, 0, 0]
    NEG_X = [-1, 0, 0]
    Y = [0, 1, 0]
    NEG_Y = [0, -1, 0]


class SimulationFields(Enum):
    """
    An enumeration that provides string constants for accessing fields in a simulation configuration file.

    The fields include metadata about the simulation, parameters of the simulation and dimensions of the material used.

    Attributes:
    -----------
    SIMULATION_METADATA : str
        The key to access metadata about the simulation.
    TITLE : str
        The key to access the title of the simulation.
    DESCRIPTION : str
        The key to access the description of the simulation.
    DATE : str
        The key to access the date the simulation was created.
    AUTHOR : str
        The key to access the name of the author who created the simulation.
    SIMULATION_PARAMETERS : str
        The key to access parameters of the simulation.
    USE_TWO_MATERIAL_LAYERS : str
        The key to access the information whether two material layers are used in the simulation.
    INTEGRATION_TIME_LIMIT : str
        The key to access the time limit for integration in the simulation.
    TOTAL_TIMESTEPS : str
        The key to access the total number of time steps in the simulation.
    hardening_model : str
        The key to access the type of hardening model chosen for the simulation.
    FIELD_INPUT_FILE : str
        The key to access the name of the input file for the simulation field.
    MATERIAL_DIMENSIONS : str
        The key to access the dimensions of the material used in the simulation.
    THICKNESS : str
        The key to access the thickness of the material layer.
    LENGTH : str
        The key to access the length of the material.
    WIDTH : str
        The key to access the width of the material.
    LAYER_1 : str
        The key to access the definition of the first outside layer
    DIRECTION: str
        The key to access the direction of the layer.
    """
    SIMULATION_METADATA = "simulation_metadata"
    TITLE = "title"
    DESCRIPTION = "description"
    DATE = "date"
    AUTHOR = "author"

    SIMULATION_PARAMETERS = "simulation_parameters"
    USE_TWO_MATERIAL_LAYERS = "use_two_material_layers"
    INTEGRATION_TIME_LIMIT = "integration_time_limit"
    TOTAL_TIMESTEPS = "total_timesteps"
    FIELD_INPUT_FILE = "field_input_file"

    MATERIAL_DIMENSIONS = "material_dimensions"
    LENGTH = "length"
    WIDTH = "width"

    LAYER_1 = "layer-1"
    THICKNESS = "thickness"
    DIRECTION = "direction"
    LAYER_MATERIAL = "material"

    MESH_RESOLUTION = "mesh_resolution"
    RESOLUTION_X = "x"
    RESOLUTION_Y = "y"

    FINITE_ELEMENT_DEGREES = "finite_element_degrees"
    FEM_DEG_U = "u"
    FEM_DEG_STRESS = "stress"

    MATERIAL_MODEL = "material_model"
    MATERIAL = "material"
    HARDENING_MODEL = "hardening_model"

    BOUNDARY_CONDITIONS = "boundary_conditions"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class SimulationConfig:
    """
    A class used to represent the simulation configuration which encapsulates
    the parameters required for a simulation.

    The configuration parameters are loaded from a JSON file. The JSON keys are
    standardized and accessed via an enumeration, `SimulationFields`.

    This class uses Python properties to provide a clean, Pythonic interface
    for accessing configuration data.

    Attributes
    ----------
    _config : dict
        A dictionary to store the configuration parameters read from the JSON file.

    Properties
    ----------
    title : str
        Title of the simulation (getter only).
    description : str
        Description of the simulation (getter only).
    date : str
        Date when the simulation configuration was created (getter only).
    author : str
        Author who created the simulation configuration (getter only).
    use_two_material_layers : bool
        Specifies whether to use two material layers in the simulation (getter only).
    integration_time_limit : int
        Specifies the endpoint for the time integration in the simulation (getter only).
    total_timesteps : int
        Specifies the total number of timesteps for the simulation (getter only).
    hardening_model : str
        Specifies the type of hardening model to be used in the simulation (getter only).
    field_input_file : str
        Specifies the file from which to read the simulation field input data (getter only).
    layer_1_thickness : float
        Specifies the thickness of the material layer in the simulation (getter only).
    length : float
        Specifies the length of the material in the simulation (getter only).
    width : float
        Specifies the width of the material in the simulation (getter only).
    layer_1_direction : array-like (3D)
        Normal vector in which the first outside layer is attached to.

    Methods
    -------
    __init__(self, config_file: str)
        Initializes a SimulationConfig object by loading a JSON file and populating the _config dictionary.
    _load_config(config_file)
        Reads the JSON file, checks for file existence and valid JSON format and returns the configuration dictionary.
    """

    def __init__(self, config_file: str):
        """Initializes a SimulationConfig object."""
        self._config = self._load_config(config_file)

    @staticmethod
    def _load_config(config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        with open(config_file) as json_file:
            try:
                config = json.load(json_file)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in configuration file '{config_file}'.")
        return config

    def get(self, key: str, subkey: str = None, subsubkey: str = None, default=None) -> Any:
        """
        Get the value of the given property.

        Parameters:
        ----------
        key : str
            The first-level key to get.
        subkey : str, optional
            The second-level key to get. Defaults to None.
        subsubkey : str, optional
            The third-level key to get. Defaults to None.
        default : Any, optional
            The value to return if the property does not exist.

        Returns:
        -------
        Any
            The value of the property, if it exists. Otherwise, the default value.
        """
        value = self._config.get(key, default)
        if subkey is not None and isinstance(value, dict):
            value = value.get(subkey, default)
        if subsubkey is not None and isinstance(value, dict):
            value = value.get(subsubkey, default)
        return value

    @property
    def title(self) -> Any:
        return self.get(SimulationFields.SIMULATION_METADATA.value, SimulationFields.TITLE.value, None)

    @property
    def description(self) -> Any:
        return self.get(SimulationFields.SIMULATION_METADATA.value, SimulationFields.DESCRIPTION.value, None)

    @property
    def date(self) -> Any:
        return self.get(SimulationFields.SIMULATION_METADATA.value, SimulationFields.DATE.value, None)

    @property
    def author(self) -> Any:
        return self.get(SimulationFields.SIMULATION_METADATA.value, SimulationFields.AUTHOR.value, None)

    @property
    def use_two_material_layers(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.USE_TWO_MATERIAL_LAYERS.value,
                        None)

    @property
    def integration_time_limit(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.INTEGRATION_TIME_LIMIT.value,
                        None)

    @property
    def total_timesteps(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.TOTAL_TIMESTEPS.value, None)

    @property
    def hardening_model(self) -> Any:
        return self.get(SimulationFields.MATERIAL_MODEL.value, SimulationFields.HARDENING_MODEL.value, None)

    @property
    def field_input_file(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.FIELD_INPUT_FILE.value, None)

    @property
    def length(self) -> Any:
        return self.get(SimulationFields.MATERIAL_DIMENSIONS.value, SimulationFields.LENGTH.value, None)

    @length.setter
    def length(self, value: Any) -> None:
        self._config[SimulationFields.MATERIAL_DIMENSIONS.value][SimulationFields.LENGTH.value] = value

    @property
    def width(self) -> Any:
        return self.get(SimulationFields.MATERIAL_DIMENSIONS.value, SimulationFields.WIDTH.value, None)

    @width.setter
    def width(self, value: Any) -> None:
        self._config[SimulationFields.MATERIAL_DIMENSIONS.value][SimulationFields.WIDTH.value] = value

    @property
    def layer_1_thickness(self) -> Any:
        return self.get(SimulationFields.LAYER_1.value, SimulationFields.THICKNESS.value, None)

    @property
    def layer_1_direction(self) -> Any:
        direction = self.get(SimulationFields.LAYER_1.value, SimulationFields.DIRECTION.value, None)
        return Direction[direction.upper()].value if direction else [0, 0, 0]

    @property
    def mesh_resolution_x(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.MESH_RESOLUTION.value,
                        SimulationFields.RESOLUTION_X.value, None)

    @property
    def mesh_resolution_y(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.MESH_RESOLUTION.value,
                        SimulationFields.RESOLUTION_Y.value, None)

    @property
    def finite_element_degree_u(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.FINITE_ELEMENT_DEGREES.value,
                        SimulationFields.FEM_DEG_U.value, None)

    @property
    def finite_element_degree_stress(self) -> Any:
        return self.get(SimulationFields.SIMULATION_PARAMETERS.value, SimulationFields.FINITE_ELEMENT_DEGREES.value,
                        SimulationFields.FEM_DEG_STRESS.value, None)

    @property
    def material(self) -> Any:
        return self.get(SimulationFields.MATERIAL_MODEL.value, SimulationFields.MATERIAL.value, None)

    @property
    def layer_material(self) -> Any:
        return self.get(SimulationFields.LAYER_1.value, SimulationFields.LAYER_MATERIAL.value, None)

    @property
    def boundary_condition_left(self) -> Any:
        return self.get(SimulationFields.BOUNDARY_CONDITIONS.value, SimulationFields.LEFT.value, None)

    @property
    def boundary_condition_right(self) -> Any:
        return self.get(SimulationFields.BOUNDARY_CONDITIONS.value, SimulationFields.RIGHT.value, None)

    @property
    def boundary_condition_top(self) -> Any:
        return self.get(SimulationFields.BOUNDARY_CONDITIONS.value, SimulationFields.TOP.value, None)

    @property
    def boundary_condition_bottom(self) -> Any:
        return self.get(SimulationFields.BOUNDARY_CONDITIONS.value, SimulationFields.BOTTOM.value, None)
