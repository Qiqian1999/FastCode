
output_size = 3
output = 'input_layer_3x3'

with open(output, 'w') as fw:
    for i in range(output_size):
        for j in range(output_size):
            if j == 0:
                fw.write('%d' % j)
            else:
                fw.write(' %d' % j)
        fw.write('\n')
