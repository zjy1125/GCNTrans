def get_distance_map(filename):
    with open('/atom/' + filename) as f:
        all_aa_ca_coordinate = {}
        for line in f:
            content = line.strip().split(' ')
            if content[2]=='CA':
                coordinate = (content[6],content[7],content[8])
                all_aa_ca_coordinate[content[5]] = coordinate
    distance_map = []
    distance_map_tuple = []
    for aa1,cor1 in all_aa_ca_coordinate.items():
        row_dis = []
        row_dis_tuple = {}
        for aa2,cor2 in all_aa_ca_coordinate.items():
            aa1_aa2 = (aa1,aa2)
            cor1 = list(map(float, cor1))
            cor2 = list(map(float, cor2))
            import numpy as np
            cor1 = np.array(cor1)
            cor2 = np.array(cor2)
            dis = np.linalg.norm(cor1-cor2)
            dis = "%.1f" % dis
            dis = float(dis)
            row_dis.append(dis)
            row_dis_tuple[aa1_aa2] = dis
        distance_map.append(row_dis)
        distance_map_tuple.append(row_dis_tuple)
    return distance_map,distance_map_tuple
  def get_array(filename):
    dis_map = np.load('/dis_map/'+filename)
    array_a = []
    for raw in dis_map:
        raw_new = []
        for column in raw:
            if column>10:
                raw_new.append(0)
            else:
                raw_new.append(1)
        array_a.append(raw_new)
    array_a = np.array(array_a)
    np.save('/graph/'+file[:-8]+'.npy', array_a)
def get_edges_unordered(file):
    try:
        graph_data = np.load('/graph/'+file)
        edges_unordered = []
        rows,cols = graph_data.shape 
        for i in range(rows):
            for j in range(cols):
                a=graph_data[i][j] 
                if a==1:
                    edge = [i,j]
                    edges_unordered.append(edge)
        edges_unordered = np.array(edges_unordered)
        np.save('/edges_unordered/'+file, edges_unordered)
    except Exception as e:
        print(f"文件 {file} 处理时出错: {e}")


