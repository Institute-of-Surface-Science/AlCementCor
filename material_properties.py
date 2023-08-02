import json
from enum import Enum
from typing import Any, Dict
from jsonschema import validate, ValidationError


class Property(Enum):
    YOUNGS_MODULUS = "Youngs_modulus"
    POISSONS_RATIO = "Poissons_ratio"
    YIELD_STRENGTH = "Yield_strength"
    SHEAR_MODULUS = "Shear_modulus"
    FIRST_LAME_PARAMETER = "First_Lame_parameter"
    TANGENT_MODULUS = "Tangent_modulus"
    LINEAR_ISOTROPIC_HARDENING = "Linear_isotropic_hardening"
    NONLINEAR_LUDWIK_PARAMETER = "Nonlinear_Ludwik_parameter"
    EXPONENT_LUDWIK = "Exponent_Ludwik"
    SWIFT_EPSILON0 = "Swift_epsilon0"
    EXPONENT_SWIFT = "Exponent_Swift"


class MaterialProperties:
    """
    Class representing the properties of a material.

    Attributes:
    ----------
    file_name : str
        The name of the JSON file containing the material properties.
    material : str
        The specific material to load properties for.

    Methods:
    -------
    get_property(property_name : str) -> Union[int, float]
        Gets the value of the given property.

    Properties:
    ----------
    youngs_modulus
    poisson_ratio
    yield_strength
    shear_modulus
    first_lame_parameter
    tangent_modulus
    linear_isotropic_hardening
    nonlinear_ludwik_parameter
    exponent_ludwik
    swift_epsilon0
    exponent_swift

    Raises:
    ------
    KeyError:
        If the given property name is not in the material properties dictionary.
    ValueError:
        If the provided JSON file does not follow the expected structure/schema.
    FileNotFoundError:
        If the provided JSON file or schema file does not exist.

    """

    def __init__(self, json_file: str, material: str) -> None:
        """
        Constructs all the necessary attributes for the material properties object.

        Parameters:
        ----------
        file_name : str
            The name of the JSON file containing the material properties.
        material : str
            The specific material to load properties for.
        """
        self.properties = self.load_and_validate(json_file, material)
        self.material = material

    @staticmethod
    def load_and_validate(json_file: str, material: str) -> Dict[str, Any]:
        schema = MaterialProperties.load_schema()
        all_materials = MaterialProperties.load_materials(json_file)
        MaterialProperties.validate(all_materials, schema)

        if material not in all_materials:
            raise ValueError(f"Material {material} not found in the JSON file.")

        return MaterialProperties.calculate_properties(all_materials[material]["properties"])

    @staticmethod
    def load_schema() -> Dict[str, Any]:
        with open('material_properties.schema') as f:
            return json.load(f)

    @staticmethod
    def load_materials(json_file: str) -> Dict[str, Any]:
        with open(json_file) as file:
            return json.load(file)

    @staticmethod
    def validate(materials: Dict[str, Any], schema: Dict[str, Any]) -> None:
        try:
            validate(instance=materials, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e.message}")

    @staticmethod
    def calculate_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        E = properties.get(Property.YOUNGS_MODULUS.value)
        nu = properties.get(Property.POISSONS_RATIO.value)

        if E is not None and nu is not None:
            properties.setdefault(Property.SHEAR_MODULUS.value, E / (2.0 * (1 + nu)))
            properties.setdefault(Property.FIRST_LAME_PARAMETER.value, E * nu / ((1 + nu) * (1 - 2 * nu)))

        if E is not None:
            properties.setdefault(Property.TANGENT_MODULUS.value, E / 100.0)

        Et = properties.get(Property.TANGENT_MODULUS.value)
        if E is not None and Et is not None:
            properties.setdefault(Property.LINEAR_ISOTROPIC_HARDENING.value, E * Et / (E - Et))

        H = properties.get(Property.LINEAR_ISOTROPIC_HARDENING.value)
        if H is not None:
            properties.setdefault(Property.NONLINEAR_LUDWIK_PARAMETER.value, 0.9 * H)

        return properties

    def get(self, key: str, default=None) -> Any:
        """
        Get the value of the given property.

        Parameters:
        ----------
        key : str
            The name of the property to get.
        default : Any, optional
            The value to return if the property does not exist.

        Returns:
        -------
        Any
            The value of the property, if it exists. Otherwise, the default value.
        """
        return self.properties.get(key, default)

    @property
    def youngs_modulus(self) -> Any:
        return self.get(Property.YOUNGS_MODULUS.value, None)

    @property
    def poisson_ratio(self) -> Any:
        return self.get(Property.POISSONS_RATIO.value, None)

    @property
    def yield_strength(self) -> Any:
        return self.get(Property.YIELD_STRENGTH.value, None)

    @property
    def shear_modulus(self) -> Any:
        return self.get(Property.SHEAR_MODULUS.value, None)

    @property
    def first_lame_parameter(self) -> Any:
        return self.get(Property.FIRST_LAME_PARAMETER.value, None)

    @property
    def tangent_modulus(self) -> Any:
        return self.get(Property.TANGENT_MODULUS.value, None)

    @property
    def linear_isotropic_hardening(self) -> Any:
        return self.get(Property.LINEAR_ISOTROPIC_HARDENING.value, None)

    @property
    def nonlinear_ludwik_parameter(self) -> Any:
        return self.get(Property.NONLINEAR_LUDWIK_PARAMETER.value, None)

    @property
    def exponent_ludwik(self) -> Any:
        return self.get(Property.EXPONENT_LUDWIK.value, None)

    @property
    def swift_epsilon0(self) -> Any:
        return self.get(Property.SWIFT_EPSILON0.value, None)

    @property
    def exponent_swift(self) -> Any:
        return self.get(Property.EXPONENT_SWIFT.value, None)
