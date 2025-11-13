# Create configuration
from jaxtrace.gpu import GPUForestConfig

config = GPUForestConfig()  # 32 blocks by default
print(config)

# Create forest grid
from jaxtrace.gpu.forest import create_regular_forest_grid
import numpy as np

bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0])
blocks = create_regular_forest_grid(bounds, (4, 4, 2))

print(f"Created {len(blocks)} blocks")
print(f"Block 0: neighbors={blocks[0].neighbors}")

# Visualize
from jaxtrace.gpu.forest import visualize_forest_blocks

visualize_forest_blocks(blocks, save_path='forest.png')