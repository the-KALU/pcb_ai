import os
import json
from pathlib import Path

def extract_components_from_pcb(pcb_file_path):
    components = []
    with open(pcb_file_path, 'r') as f:
        for line in f:
            if 'footprint' in line.lower() or 'module' in line.lower():
                components.append(line.strip())
    return components

def extract_nets_from_sch(sch_file_path):
    nets = []
    with open(sch_file_path, 'r') as f:
        for line in f:
            if 'Net' in line:
                nets.append(line.strip())
    return nets

def preprocess_project(project_path):
    project_name = project_path.name
    pcb_file = next(project_path.glob("*.kicad_pcb"))
    sch_file = next(project_path.glob("*.sch"))

    components = extract_components_from_pcb(pcb_file)
    nets = extract_nets_from_sch(sch_file)

    prompt = f"Design a PCB for {project_name.replace('_', ' ')} using standard components."
    response = {
        "components": components[:10],  # limit for now
        "nets": nets[:10],
        "layout_format": "KiCad",
        "summary": f"This design includes {len(components)} components and {len(nets)} nets."
    }

    return {
        "prompt": prompt,
        "response": response
    }

# Loop through projects
if __name__ == "__main__":
    project_dir = Path("../pcb_projects/")
    output_data = []

    for project in project_dir.iterdir():
        if project.is_dir():
            try:
                data = preprocess_project(project)
                output_data.append(data)
            except Exception as e:
                print(f"Failed on {project.name}: {e}")

    with open("dataset.json", "w") as f:
        json.dump(output_data, f, indent=2)
