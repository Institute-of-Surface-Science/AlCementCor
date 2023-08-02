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

## Contributing

We welcome contributions to this repository. If you wish to contribute, please follow the standard Fork & Pull Request workflow.

## License

This project is licensed under the MIT License. See `LICENSE` file in the repository for more details.