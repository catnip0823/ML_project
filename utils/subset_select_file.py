
def subset_select(data):

    set_size = {1: [data[:, :1], data[:, 1:2], data[:, 2:3], data[:, 3:4], data[:, 4:5], data[:, 5:6], data[:, 6:]]}


    temp_list = []
    for i in range(7):
        for j in range(7):
            if i < j:
                temp_list.append(data[:, [i, j]])

    set_size[2] = temp_list

    temp_list = []
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if i < j < k:
                    temp_list.append(data[:, [i, j, k]])
    set_size[3] = temp_list

    temp_list = []
    for i in range(7):
        for j in range(7):
            for k in range(7):
                for l in range(7):
                    if i < j < k < l:
                        temp_list.append(data[:, [i, j, k, l]])
    set_size[4] = temp_list

    temp_list = []
    for i in range(7):
        for j in range(7):
            for k in range(7):
                for l in range(7):
                    for m in range(7):
                        if i < j < k < l < m:
                            temp_list.append(data[:, [i, j, k, l, m]])
    set_size[5] = temp_list
    set_size[6] = [data[:, [0, 1, 2, 3, 4, 5]], data[:, [0, 6, 2, 3, 4, 5]], data[:, [0, 1, 6, 3, 4, 5]], data[:, [0, 1, 2, 6, 4, 5]], data[:, [0, 1, 2, 3, 6, 5]], data[:, [0, 1, 2, 3, 4, 6]], data[:, [6, 1, 2, 3, 4, 5]]]
    set_size[7] = [data]
    return set_size


