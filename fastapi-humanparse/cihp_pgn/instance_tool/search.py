import numpy as np
from scipy import ndimage


def search(parsing_map, input_edge, edge_thresh):
    threshold = 1
    padding = 10

    row, col = parsing_map.shape

    edge_map = (input_edge >= edge_thresh).astype(float)
    edge_map = ndimage.binary_closing(edge_map).astype(float)
    edge_map = ndimage.binary_thin(edge_map).astype(float)

    map_vertical = np.zeros([row, col])
    map_horizontal = np.zeros([row, col])

    num_ins = 0
    for c in range(col):
        for r in range(row):
            if edge_map[r, c] == threshold:
                if row - r < padding:
                    for a_r in range(r + 1, row):
                        edge_map[a_r, c] = threshold
                continue
            if parsing_map[r, c] != 0:
                if c - 1 > 0 and np.sum(map_vertical[:, c - 1]) == 0 and np.sum(map_vertical[:, c]) == 0:
                    num_ins += 1
                    map_vertical[r, c] = num_ins
                else:
                    if r - 1 > 0 and map_vertical[r - 1, c] != 0:
                        map_vertical[r, c] = map_vertical[r - 1, c]
                    if map_vertical[r, c] == 0:
                        num_ins += 1
                        map_vertical[r, c] = num_ins

    num_ins = 0
    for r in range(row):
        for c in range(col):
            if edge_map[r, c] == threshold:
                if col - c < padding:
                    for a_c in range(c + 1, col):
                        edge_map[r, a_c] = threshold
                continue
            if parsing_map[r, c] != 0:
                if r - 1 > 0 and np.sum(map_horizontal[r - 1, :]) == 0 and np.sum(map_horizontal[r, :]) == 0:
                    num_ins += 1
                    map_horizontal[r, c] = num_ins
                else:
                    if c - 1 > 0 and map_horizontal[r, c - 1] != 0:
                        map_horizontal[r, c] = map_horizontal[r, c - 1]
                    if map_horizontal[r, c] == 0:
                        num_ins += 1
                        map_horizontal[r, c] = num_ins

    map_combine, num_instance = bfs(map_horizontal, map_vertical)

    refined_map = region_merge(map_combine, parsing_map, num_instance, padding)
    num_refined_instance = len(np.unique(refined_map)) - 1
    while num_refined_instance != num_instance:
        num_instance = num_refined_instance
        refined_map = region_merge(refined_map, parsing_map, num_instance, 0)
        num_refined_instance = len(np.unique(refined_map)) - 1

    return map_horizontal, map_vertical, map_combine, refined_map
