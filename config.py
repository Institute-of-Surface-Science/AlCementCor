import json
import os
from enum import Enum


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
    CHOSEN_HARDENING_MODEL : str
        The key to access the type of hardening model chosen for the simulation.
    FIELD_INPUT_FILE : str
        The key to access the name of the input file for the simulation field.
    MATERIAL_DIMENSIONS : str
        The key to access the dimensions of the material used in the simulation.
    LAYER_THICKNESS : str
        The key to access the thickness of the material layer.
    LENGTH : str
        The key to access the length of the material.
    WIDTH : str
        The key to access the width of the material.
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
    CHOSEN_HARDENING_MODEL = "chosen_hardening_model"
    FIELD_INPUT_FILE = "field_input_file"

    MATERIAL_DIMENSIONS = "material_dimensions"
    LAYER_THICKNESS = "layer_thickness"
    LENGTH = "length"
    WIDTH = "width"


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
    chosen_hardening_model : str
        Specifies the type of hardening model to be used in the simulation (getter only).
    field_input_file : str
        Specifies the file from which to read the simulation field input data (getter only).
    layer_thickness : float
        Specifies the thickness of the material layer in the simulation (getter only).
    length : float
        Specifies the length of the material in the simulation (getter only).
    width : float
        Specifies the width of the material in the simulation (getter only).

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

    @property
    def title(self):
        return self._config[SimulationFields.SIMULATION_METADATA.value][SimulationFields.TITLE.value]

    @property
    def description(self):
        return self._config[SimulationFields.SIMULATION_METADATA.value][SimulationFields.DESCRIPTION.value]

    @property
    def date(self):
        return self._config[SimulationFields.SIMULATION_METADATA.value][SimulationFields.DATE.value]

    @property
    def author(self):
        return self._config[SimulationFields.SIMULATION_METADATA.value][SimulationFields.AUTHOR.value]

    @property
    def use_two_material_layers(self):
        return self._config[SimulationFields.SIMULATION_PARAMETERS.value][
            SimulationFields.USE_TWO_MATERIAL_LAYERS.value]

    @property
    def integration_time_limit(self):
        return self._config[SimulationFields.SIMULATION_PARAMETERS.value][SimulationFields.INTEGRATION_TIME_LIMIT.value]

    @property
    def total_timesteps(self):
        return self._config[SimulationFields.SIMULATION_PARAMETERS.value][SimulationFields.TOTAL_TIMESTEPS.value]

    @property
    def chosen_hardening_model(self):
        return self._config[SimulationFields.SIMULATION_PARAMETERS.value][SimulationFields.CHOSEN_HARDENING_MODEL.value]

    @property
    def field_input_file(self):
        return self._config[SimulationFields.SIMULATION_PARAMETERS.value][SimulationFields.FIELD_INPUT_FILE.value]

    @property
    def layer_thickness(self):
        return self._config[SimulationFields.MATERIAL_DIMENSIONS.value][SimulationFields.LAYER_THICKNESS.value]

    @property
    def length(self):
        return self._config[SimulationFields.MATERIAL_DIMENSIONS.value][SimulationFields.LENGTH.value]

    @property
    def width(self):
        return self._config[SimulationFields.MATERIAL_DIMENSIONS.value][SimulationFields.WIDTH.value]
