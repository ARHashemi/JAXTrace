import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configuration settings
TOTAL_STEPS = 8000
CYCLE_START = 34
CYCLE_END = 199
TIME_START = 0.108507
TIME_STEP = 0.000363  # Calculated from your data: 0.10887 - 0.108507

# Create the core XML structure
root = ET.Element("VTKFile", type="Collection", version="0.1")
collection = ET.SubElement(root, "Collection")

for i in range(TOTAL_STEPS):
    # Calculate the continuous matching timeline for your point cloud
    current_time = TIME_START + (i * TIME_STEP)

    # Determine which file index to pull from based on cycling rules
    if i <= CYCLE_START:
        file_index = i
    else:
        # Loop strictly within the 34 to 199 range
        cycle_length = CYCLE_END - CYCLE_START + 1
        file_index = CYCLE_START + ((i - CYCLE_START) % cycle_length)

    # FIX: Changed "post//" to "post/" to prevent ParaView from duplicating the directory
    ET.SubElement(
        collection,
        "DataSet",
        timestep=f"{current_time:.6f}",
        group="",
        part=str(i),
        file=f"post/B4_{file_index}.pvtu"
    )

# Format the XML layout to be clean and readable
xml_string = ET.tostring(root, encoding="utf-8")
parsed_xml = minidom.parseString(xml_string)
pretty_xml = parsed_xml.toprettyxml(indent="  ")

# Save to your new master PVD file
with open("mesh_cyclic_8000.pvd", "w", encoding="utf-8") as f:
    f.write(pretty_xml)

print("Successfully generated 'mesh_cyclic_8000.pvd' with fixed file paths!")
