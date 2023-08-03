import inspect
import os
from tabulate import tabulate
from typing import List
from AlCementCor.config import SimulationConfig
from AlCementCor.material_properties import MaterialProperties


def summarize_and_print_config(simulation_config: SimulationConfig,
                               materials: List[MaterialProperties]) -> None:
    terminal_width = os.get_terminal_size().columns

    logo = """
        ___    ________                          __  ______          
       /   |  / / ____/__  ____ ___  ___  ____  / /_/ ____/___  _____
      / /| | / / /   / _ \/ __ `__ \/ _ \/ __ \/ __/ /   / __ \/ ___/
     / ___ |/ / /___/  __/ / / / / /  __/ / / / /_/ /___/ /_/ / /    
    /_/  |_/_/\____/\___/_/ /_/ /_/\___/_/ /_/\__/\____/\____/_/     
    by Sven Berger and Aravinth Ravikumar
    Helmholtz-Center hereon, Institute of Surface Science 
    Copyright 2023
        """

    print(logo)

    # Print simulation configuration
    print("\nSimulation Configuration:")
    print("-" * terminal_width)

    # Dynamically get the properties of the SimulationConfig class
    sim_config_props = [attr for attr in dir(simulation_config) if
                        not callable(getattr(simulation_config, attr)) and not attr.startswith("_")]

    config_dict = {prop: getattr(simulation_config, prop) for prop in sim_config_props}

    for property, value in config_dict.items():
        print(f"  {property}: {value}")

    # Print a line that spans the entire width
    print("-" * terminal_width)

    # Print material properties
    print("\nMaterial Properties:")
    print("-" * terminal_width)

    # Dynamically get the properties of the MaterialProperties class
    material_props = [attr for attr in dir(MaterialProperties) if
                      not callable(getattr(MaterialProperties, attr)) and not attr.startswith("_")]
    headers = ["Property"] + [material.material for material in materials]
    rows = []
    for prop in material_props:
        row = [prop]  # Start with the property name
        for material in materials:
            value = getattr(material, prop, None)
            row.append(
                value if value is not None else "N/A")  # Append each property value to the row, "N/A" if not exist
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="pipe", floatfmt=".2f", missingval="N/A"))
    print("-" * terminal_width)
