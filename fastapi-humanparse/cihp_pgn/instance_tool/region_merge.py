import numpy as np
from skimage.morphology import remove_small_objects

human_regions = []
human_labels = []
num_human = 0
part_regions = []
num_part = 0

for i in range(1, num_instance+1):
        region_i = (map_combine == i)
        if np.sum(region_i) == 0:
            continue
        parsing_i = parsing_map[region_i]
        num_label = np.unique(parsing_i)
        if len(num_label) > 1 and np.sum(region_i) > 30:
            num_human += 1
            human_regions.append(region_i)
            human_labels.append(i)
        else:
            num_part += 1
            part_regions.append(region_i)

for i in range(num_part):
    part_i = part_regions[i]
    boundary_i = np.array(morphology.remove_small_objects(part_i.astype(bool), connectivity=1), dtype=int)
    ele_row, ele_col = np.where(boundary_i > 0)
    is_merge = 0
    for e in range(len(ele_row)):
        if is_merge == 1:
            break
        rr = ele_row[e]
        cc = ele_col[e]
        cur_label = map_combine[rr, cc]
        if is_merge == 0 and rr - 2 > 0 and map_combine[rr-2, cc] > 0 and map_combine[rr-2, cc] != cur_label:
            for p in range(i+1, num_part):
                part_p = part_regions[p]
                if part_p[rr-2, cc] > 0:
                    is_merge = 1
                    map_combine[part_p > 0] = cur_label
                    part_regions[p] = (part_i + part_p) > 0
                    break
            for h in range(num_human):
                human_i = human_regions[h]
                if human_i[rr-2, cc] > 0:
                    is_merge = 1
                    map_combine[part_i > 0] = human_labels[h]
                    break
        elif is_merge == 0 and cc - 2 > 0 and map_combine[rr, cc-2] > 0 and map_combine[rr, cc-2] != cur_label:
            for p in range(i+1, num_part):
                part_p = part_regions[p]
                if part_p[rr, cc-2] > 0:
                    is_merge = 1
                    map_combine[part_p > 0] = cur_label
                    part_regions[p] = (part_i + part_p) > 0
                    break
            for h in range(num_human):
                human_i = human_regions[h]
                if human_i[rr, cc-2] > 0:
                    is_merge = 1
                    map_combine[part_i > 0] = human_labels[h]
                    break
        elif is_merge == 0 and rr + 2 <= row and map_combine[rr+2, cc] > 0 and map_combine[rr+2, cc] != cur_label:
            for p in range(i+1, num_part):
                part_p = part_regions[p]
                if part_p[rr+2, cc] > 0:
                    is_merge = 1
                    map_combine[part_p > 0] = cur_label
                    part_regions[p] = (part_i + part_p) > 0
                    break
            for h in range(num_human):
                human_i = human_regions[h]
                if human_i[rr+2, cc] > 0:
                    is_merge = 1
                    map_combine[part_i > 0] = human_labels[h]
                    break
        elif is_merge == 0 and cc + 2 <= col and map_combine[rr, cc+2] > 0 and map_combine[rr, cc+2] != cur_label:
            for p in range(i+1, num_part):
                part_p = part_regions[p]
                if part_p[rr, cc+2] > 0:
                    is_merge = 1
                    map_combine[part_p > 0] = cur_label
                    part_regions[p] = (part_i + part_p) > 0
                    break
            for h in range(num_human):
                human_i = human_regions[h]
                if human_i[rr, cc+2] > 0:
                    is_merge = 1
                    map_combine[part_i > 0] = human_labels[h]
                    break

if padding > 0:
    for r in range(row):
        if map_combine[r, col-padding] != 0:
            for c in range(col-padding-1, col):
                if map_combine[r,c] == 0:
                    map_combine[r,c] = map_combine[r,c-1]
        if map_combine[r, padding] != 0:
            for c in range(padding-1, -1, -1):
                if map_combine[r,c] == 0:
                    map_combine[r,c] = map_combine[r,c+1]
    for c in range(col):
        if map_combine[row-padding, c] != 0:
            for r in range(row-padding-1, row):
                if map_combine[r,c] == 0:
                    map_combine[r,c] = map_combine[r-1,c]
        if map_combine[padding,c] != 0:
            for r in range(padding-1, -1, -1):
                if map_combine[r,c] == 0:
                    map_combine[r,c] = map_combine[r+1,c]

refined_map = map_combine