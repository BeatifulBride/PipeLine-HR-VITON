import numpy as np


def bfs(instance_map_horizontal, instance_map_vertical):
    global horizontal_map, vertical_map, marked, queue_sum, row, col

    horizontal_map = instance_map_horizontal
    vertical_map = instance_map_vertical

    row, col = instance_map_horizontal.shape
    instance_map = np.zeros((row, col))
    marked = np.zeros((row*col), dtype=int)
    queue_sum = []

    for i in range(row):
        for j in range(col):
            if marked[(i-1)*col+j] == 0 and horizontal_map[i,j] != 0 and vertical_map[i,j] != 0:
                bfs_search((i-1)*col+j)

    num_instance = len(queue_sum)

    for i in range(len(queue_sum)):
        region = queue_sum[i]
        for j in range(len(region)):
            x = (region[j]-1) // col
            y = (region[j]-1) % col
            instance_map[x,y] = i

    return instance_map, num_instance


def bfs_search(start):
    global marked, queue_sum

    queue = [start]
    region_queue = []
    while queue:
        node = queue.pop(0)
        if marked[node] == 0:
            marked[node] = 1
            region_queue.append(node)

            neighbours = find_neighbour(node)
            for neighbour in neighbours:
                queue.append(neighbour)

    if region_queue:
        point_num = len(region_queue)
        index = 0
        for q in range(len(queue_sum)):
            point_q = len(queue_sum[q])
            if point_num < point_q:
                index += 1
            else:
                break
        if queue_sum:
            for p in range(len(queue_sum), index, -1):
                queue_sum[p] = queue_sum[p-1]
        queue_sum.insert(index, region_queue)


def find_neighbour(node):
    global marked, horizontal_map, vertical_map, col, row

    i = node // col
    j = node % col
    neighbours = []

    x = i-1
    y = j
    if x >= 0 and vertical_map[x,y] == vertical_map[i,j] and marked[x*col+y] == 0 and vertical_map[x,y] != 0:
        neighbours.append(x*col+y)

    x = i+1
    y = j
    if x < row and vertical_map[x,y] == vertical_map[i,j] and marked[x*col+y] == 0 and vertical_map[x,y] != 0:
        neighbours.append(x*col+y)

    x = i
    y = j-1
    if y >= 0 and horizontal_map[x,y] == horizontal_map[i,j] and marked[x*col+y] == 0 and horizontal_map[x,y] != 0:
        neighbours.append(x*col+y)

    x = i
    y = j+1
    if y < col and horizontal_map[x,y] == horizontal_map[i,j] and marked[x*col+y] == 0 and horizontal_map[x,y] != 0:
        neighbours.append(x*col+y)

    return neighbours
