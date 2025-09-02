import numpy as np
thetas=np.random.rand(784,25)*0.05
print(type(thetas))

def thetas_rolled(unrolled_thetas,layers):
    num_cengshu=len(layers)
    roll_thetas={}#是因为考虑到每一层对应一个矩阵，大概的数据结构应该是：int 对应的index，然后矩阵对应的其值

    layers_start_position=0
    for layers_index in range(num_cengshu-1):
        in_count=roll_thetas[layers_index]
        out_count=roll_thetas[layers_index+1]

        #计算矩阵的大小：
        matrix_width=in_count+1
        matrix_height=out_count
        matrix_volume=matrix_width*matrix_height

        start_index=layers_start_position
        end_index=start_index+matrix_volume
        layer_unrolled_thetas=unrolled_thetas[start_index:end_index]
        roll_thetas[layers_index]=layer_unrolled_thetas.resharp((matrix_height,matrix_width))
        layers_start_position=layers_start_position+matrix_volume
    return roll_thetas



