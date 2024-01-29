import os
import numpy as np
import cv2

colormap = np.load('pascal_seg_colormap.npy')
parsing_folder = os.path.join('../output/cihp_parsing_maps')
edge_folder = os.path.join('../output/cihp_edge_maps')
filelist = np.genfromtxt('../data/CIHP/list/val_id.txt', dtype=str)

out_dir = os.path.join('../output/cihp_human_maps')
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

edge_thresh = 0.2
human_class_id = 1

for i, img_fn in enumerate(filelist):
    print(f"num: {i+1}, {img_fn}")

parsing_map = cv2.imread(os.path.join(parsing_folder, f"{img_fn}.png"))
edge_ave_map = cv2.imread(os.path.join(edge_folder, f"{img_fn}.png"))
edge_ave_map = edge_ave_map.astype('float32') / 255

map_horizontal, map_vertical, map_combine, refined_map = search(parsing_map, edge_ave_map, edge_thresh)

out_map = np.zeros_like(refined_map)

max_ins = np.max(refined_map)
sum_map = np.zeros(max_ins)
count = 0
for ins in range(1, max_ins + 1):
    indices = np.where(refined_map == ins)
    if np.sum(indices) > 0:
        count += 1
        out_map[indices] = count
        sum_map[count - 1] = len(indices[0])

with open(os.path.join(out_dir, f"{img_fn}.txt"), 'w') as fid:
    for c in range(count):
        fid.write(f"{human_class_id}, {sum_map[c]}\n")

cv2.imwrite(os.path.join(out_dir, f"{img_fn}.png"), out_map)