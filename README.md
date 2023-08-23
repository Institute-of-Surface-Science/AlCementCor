# OpenModel Example: cement-aluminium interface damage model workflow

OpenModel example use case for damage simulation for a cement-aluminium interface damage model workflow.

## Dependencies

To use this repository, the following Python packages are required:

- Numpy
- Scipy
- Matplotlib
- jsonschema
- pandas
- sklearn
- tabulate

These packages can be installed via pip:

```sh
pip install numpy scipy matplotlib jsonschema pandas sklearn tabulate
```

## Getting Started

1. **Clone the repository:** 

```sh
git clone https://github.com/Institute-of-Surface-Science/AlCementCor.git
```

2. **Navigate to the repository folder:**

```sh
cd AlCementCor
```

3. **Run a simulation:** Use the provided example to get started.

```sh
python main.py
```

## Customizing the Simulation

You can customize your simulations by modifying the simulation configuration files and specifying different material properties and behaviours.

1. **Material Properties:** Specify the properties of the material you wish to simulate in the `material_properties.json` file.

2. **Simulation Configuration:** Adjust various parameters for the simulation in the `simulation_config.json` file. You can also use setters for some properties.

## Documentation

Please refer to the in-code comments and docstrings for more information on how to use and extend this framework.

### material_properties.json

The file represents a collection of materials with their metadata and mechanical properties.

#### Structure

Each material has a unique identifier (like `Al6082-T6`, `Aluminium-Ceramic`) which contains two main sections:

1. **metadata**: Contains information about the material.
2. **properties**: Represents the mechanical properties of the material.

#### Metadata Fields

- **material_name**: The formal or common name of the material.
- **composition**: The elemental composition or components of the material, usually given in percentage.
- **source**: Source or reference from where the material information is taken.
- **remarks**: Any additional notes or remarks regarding the material.

#### Properties Fields

- **Youngs_modulus**: Represents the Young's Modulus of the material.
- **Poissons_ratio**: Represents the Poisson's ratio.
- **Yield_strength**: Represents the yield strength.
- **Tangent_modulus**: Tangent modulus value.
- **Linear_isotropic_hardening**: Value for linear isotropic hardening.
- **Nonlinear_Ludwik_parameter**: Parameter for nonlinear Ludwik's equation (specific to some materials).
- **Exponent_Ludwik**: Exponent for Ludwik's equation (specific to some materials).
- **Swift_epsilon0**: Parameter for Swift's equation (specific to some materials).
- **Exponent_Swift**: Exponent for Swift's equation (specific to some materials).

---

**Example:**

For `Al6082-T6`:

- **Metadata**
  - **Material Name**: Aluminium 6082-T6
  - **Composition**: Aluminium: 97.4%, Magnesium: 1.2%, Silicon: 1.0%, Iron: 0.5%, Copper: 0.2%, Manganese: 0.6%, Chromium: 0.25%, Zinc: 0.2%, Titanium: 0.1%
  - **Source**: Unknown
  - **Remarks**: No additional remarks.

- **Properties**
  - **Young's Modulus**: 71000
  - **Poisson's Ratio**: 0.3
  - ... (and so on for other properties)

## Contributing

We welcome contributions to this repository. If you wish to contribute, please follow the standard Fork & Pull Request workflow.

## License

This project is licensed under the MIT License. See `LICENSE` file in the repository for more details.