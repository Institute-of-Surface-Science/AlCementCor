from tabulate import tabulate
from typing import List
import config
import material_properties as mat_prop

def summarize_and_print_config(simulation_config: config.SimulationConfig,
                               materials: List[mat_prop.MaterialProperties]) -> None:
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
    print("---------------------------------------")

    # Dynamically get the properties of the SimulationConfig class
    sim_config_props = [attr for attr in dir(simulation_config) if
                        not callable(getattr(simulation_config, attr)) and not attr.startswith("_")]

    config_dict = {prop: getattr(simulation_config, prop) for prop in sim_config_props}

    for property, value in config_dict.items():
        print(f"{property}: {value}")
    print("---------------------------------------")

    # Print material properties
    print("\nMaterial Properties:")
    print("---------------------------------------")

    # Dynamically get the properties of the MaterialProperties class
    material_props = [attr for attr in dir(mat_prop.MaterialProperties) if
                      not callable(getattr(mat_prop.MaterialProperties, attr)) and not attr.startswith("_")]
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
    print("---------------------------------------")