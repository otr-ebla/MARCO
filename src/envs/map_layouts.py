import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ProceduralMapLayout:
    """
    Generates a random indoor map layout using Binary Space Partitioning (BSP).
    Ensures a varied number of rooms, corridors, and doorways while keeping
    computational complexity low. 
    """
    def __init__(self, width=12.0, height=8.0, min_room_size=2.0, max_walls=30,
                 cell_size: float = 0.5, robot_radius: float = 0.20,
                 rng: random.Random | None = None, wall_cells: int = 1,
                 max_depth: int = 3):
        self.width = width
        self.height = height
        self.min_room_size = min_room_size
        self.max_walls = max_walls
        self.cell_size = cell_size
        self.robot_radius = robot_radius
        self._rng = rng if rng is not None else random.Random()
        # BSP recursion depth: at most 2 ** max_depth rooms (0 = one empty room).
        self.max_depth = int(max_depth)
        if self.max_depth < 0 or self.max_depth != max_depth:
            raise ValueError('max_depth must be a non-negative integer')
        self.wall_cells = int(wall_cells)
        if self.wall_cells < 0 or self.wall_cells != wall_cells:
            raise ValueError('wall_cells must be a non-negative integer (0 is legacy geometry)')
        
        self.outer_t = 0.20
        self.inner_t = 0.08
        if self.wall_cells:
            self.outer_t = self.inner_t = self.wall_cells * self.cell_size
        # Ensure door spans an integer number of cells (at least 2 cells = 1.0m)
        self.door_cells = max(2, int(np.ceil((2 * cell_size + 2 * robot_radius) / cell_size)))
        self.door_size = self.door_cells * self.cell_size
        
        self.walls = []
        self._generate_layout()
        self._pad_walls()

    def _generate_layout(self):
        # 1. Outer boundaries
        self.walls.append([-self.outer_t, -self.outer_t, self.width + self.outer_t, 0.0])
        self.walls.append([-self.outer_t, self.height, self.width + self.outer_t, self.height + self.outer_t])
        self.walls.append([-self.outer_t, -self.outer_t, 0.0, self.height + self.outer_t])
        self.walls.append([self.width, -self.outer_t, self.width + self.outer_t, self.height + self.outer_t])
        
        # 2. Recursive BSP to create rooms
        if self.wall_cells:
            self._split_grid(0, 0, int(self.width / self.cell_size),
                             int(self.height / self.cell_size), 0, set())
        else:
            self._split_space(0.0, 0.0, self.width, self.height, depth=0,
                             max_depth=self.max_depth)

    def _split_grid(self, x, y, w, h, depth, protected):
        """BSP with whole-cell wall bands and protected doorway approaches.

        Child rooms exclude the wall band. Keeping the cells immediately on
        either side of every door free prevents later splits sealing a door.
        """
        if depth >= self.max_depth:
            return
        horizontal = w < h
        if max(w, h) < 1.2 * min(w, h):
            horizontal = self._rng.choice([True, False])
        thickness = self.wall_cells
        minimum = max(1, int(np.ceil(self.min_room_size / self.cell_size)))
        along, across = (w, h) if horizontal else (h, w)
        splits = list(range(minimum, across - minimum - thickness + 1))
        self._rng.shuffle(splits)
        for split in splits:
            doors = list(range(along - self.door_cells + 1))
            self._rng.shuffle(doors)
            for door in doors:
                # Keep nonempty wall segments at least two cells long.
                if (0 < door < 2 or 0 < along - door - self.door_cells < 2):
                    continue
                def cell(a, b):
                    return (x + a, y + b) if horizontal else (x + b, y + a)
                wall = {cell(a, b) for a in range(along)
                        if not door <= a < door + self.door_cells
                        for b in range(split, split + thickness)}
                if wall & protected or not wall:
                    continue
                for a0, a1 in ((0, door), (door + self.door_cells, along)):
                    if a0 == a1:
                        continue
                    x0, y0 = cell(a0, split)
                    x1, y1 = cell(a1, split + thickness)
                    self.walls.append([v * self.cell_size for v in (x0, y0, x1, y1)])
                protected.update(cell(a, b) for a in range(door, door + self.door_cells)
                                 for b in range(split - 1, split + thickness + 1))
                if horizontal:
                    self._split_grid(x, y, w, split, depth + 1, protected)
                    self._split_grid(x, y + split + thickness, w,
                                     h - split - thickness, depth + 1, protected)
                else:
                    self._split_grid(x, y, split, h, depth + 1, protected)
                    self._split_grid(x + split + thickness, y,
                                     w - split - thickness, h, depth + 1, protected)
                return

    def _split_space(self, x, y, w, h, depth, max_depth):
        if depth >= max_depth:
            return

        # Determine split direction dynamically based on aspect ratio
        split_horizontally = w < h
        if w >= h * 1.2:
            split_horizontally = False
        elif h >= w * 1.2:
            split_horizontally = True
        else:
            split_horizontally = self._rng.choice([True, False])

        min_cells = int(round(self.min_room_size / self.cell_size))

        if split_horizontally:
            h_cells = int(round(h / self.cell_size))
            if h_cells < min_cells * 2:
                return
            y_endpoints = []
            for w_seg in self.walls:
                if abs(w_seg[2] - w_seg[0]) < 0.2:
                    cx = (w_seg[0] + w_seg[2]) / 2.0
                    if abs(cx - x) < 0.2 or abs(cx - (x + w)) < 0.2:
                        y_endpoints.append(int(round(w_seg[1] / self.cell_size)))
                        y_endpoints.append(int(round(w_seg[3] / self.cell_size)))
            
            valid_split_cells_y = []
            y_start_cell = int(round(y / self.cell_size))
            for split_cell_y_local in range(min_cells, h_cells - min_cells + 1):
                split_cell_y_global = y_start_cell + split_cell_y_local
                valid = True
                for y_ep in y_endpoints:
                    if abs(split_cell_y_global - y_ep) == 1:
                        valid = False
                        break
                if valid:
                    valid_split_cells_y.append(split_cell_y_local)
            
            if not valid_split_cells_y:
                return
                
            split_cell_y = self._rng.choice(valid_split_cells_y)
            split_y = y + split_cell_y * self.cell_size
            
            w_cells = int(round(w / self.cell_size))
            valid_door_cells = []
            for dc in range(w_cells - self.door_cells + 1):
                w1 = dc
                w2 = w_cells - dc - self.door_cells
                if (w1 == 0 or w1 >= 2) and (w2 == 0 or w2 >= 2):
                    valid_door_cells.append(dc)
            
            if not valid_door_cells:
                return
            
            door_cell = self._rng.choice(valid_door_cells)
            door_pos = x + door_cell * self.cell_size
            
            if door_pos > x:
                self.walls.append([x, split_y - self.inner_t / 2, door_pos, split_y + self.inner_t / 2])
            if (x + w) > (door_pos + self.door_size):
                self.walls.append([door_pos + self.door_size, split_y - self.inner_t / 2, x + w, split_y + self.inner_t / 2])
            
            # Recurse into the two new sub-regions with exact grid-aligned dimensions
            self._split_space(x, y, w, split_cell_y * self.cell_size, depth + 1, max_depth)
            self._split_space(x, split_y, w, (h_cells - split_cell_y) * self.cell_size, depth + 1, max_depth)
            
        else:
            w_cells = int(round(w / self.cell_size))
            if w_cells < min_cells * 2:
                return
            x_endpoints = []
            for w_seg in self.walls:
                if abs(w_seg[3] - w_seg[1]) < 0.2:
                    cy = (w_seg[1] + w_seg[3]) / 2.0
                    if abs(cy - y) < 0.2 or abs(cy - (y + h)) < 0.2:
                        x_endpoints.append(int(round(w_seg[0] / self.cell_size)))
                        x_endpoints.append(int(round(w_seg[2] / self.cell_size)))
            
            valid_split_cells_x = []
            x_start_cell = int(round(x / self.cell_size))
            for split_cell_x_local in range(min_cells, w_cells - min_cells + 1):
                split_cell_x_global = x_start_cell + split_cell_x_local
                valid = True
                for x_ep in x_endpoints:
                    if abs(split_cell_x_global - x_ep) == 1:
                        valid = False
                        break
                if valid:
                    valid_split_cells_x.append(split_cell_x_local)
            
            if not valid_split_cells_x:
                return
                
            split_cell_x = self._rng.choice(valid_split_cells_x)
            split_x = x + split_cell_x * self.cell_size
            
            h_cells = int(round(h / self.cell_size))
            valid_door_cells = []
            for dc in range(h_cells - self.door_cells + 1):
                w1 = dc
                w2 = h_cells - dc - self.door_cells
                if (w1 == 0 or w1 >= 2) and (w2 == 0 or w2 >= 2):
                    valid_door_cells.append(dc)
            
            if not valid_door_cells:
                return
                
            door_cell = self._rng.choice(valid_door_cells)
            door_pos = y + door_cell * self.cell_size
            
            if door_pos > y:
                self.walls.append([split_x - self.inner_t / 2, y, split_x + self.inner_t / 2, door_pos])
            if (y + h) > (door_pos + self.door_size):
                self.walls.append([split_x - self.inner_t / 2, door_pos + self.door_size, split_x + self.inner_t / 2, y + h])
            
            # Recurse into the two new sub-regions with exact grid-aligned dimensions
            self._split_space(x, y, split_cell_x * self.cell_size, h, depth + 1, max_depth)
            self._split_space(split_x, y, (w_cells - split_cell_x) * self.cell_size, h, depth + 1, max_depth)

    def _pad_walls(self):
        """
        Pads the walls array to a static maximum size for JAX compatibility.
        Missing walls are placed at coordinates that will never interfere with the scene.
        """
        current_walls = len(self.walls)
        if current_walls > self.max_walls:
            self.walls = self.walls[:self.max_walls]
        else:
            # Pad with dummy zero-area walls far outside the map
            padding = [[-10.0, -10.0, -10.0, -10.0] for _ in range(self.max_walls - current_walls)]
            self.walls.extend(padding)

    def get_walls(self):
        return np.array(self.walls, dtype=np.float32)


def create_map_bank(num_maps=16, seed=0, **kwargs):
    """
    Generates a bank of random maps to be cached by the JAX environment.
    Runs purely in Python during initialization to avoid JAX dynamic loop overhead.
    """
    rng = random.Random(seed)
    return [ProceduralMapLayout(rng=rng, **kwargs) for _ in range(num_maps)]


def main():
    # Test generation and visualization
    layout = ProceduralMapLayout()
    walls = layout.get_walls()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Light floor background
    floor = patches.Rectangle(
        (0, 0), layout.width, layout.height,
        facecolor="#f5f5f0", edgecolor="none", zorder=0
    )
    ax.add_patch(floor)

    # Draw each wall as a filled rectangle
    for x1, y1, x2, y2 in walls:
        # Skip padding dummy walls
        if x1 <= -5.0:
            continue
            
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            facecolor="#444444", edgecolor="black", linewidth=0.5, zorder=2
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.5, layout.width + 0.5)
    ax.set_ylim(-0.5, layout.height + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Procedural Indoor Map Layout")
    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)

    plt.tight_layout()
    plt.savefig("procedural_room_layout.png", dpi=150)
    print("Saved procedural_room_layout.png")


if __name__ == "__main__":
    main()
