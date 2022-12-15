import random

output_size = 224
output = 'input_layer_224x224'

with open(output, 'w') as fw:
    for i in range(output_size):
        for j in range(output_size):
            element = random.uniform(1, 2)
            if j == 0:
                fw.write('%.2f' % element)
            else:
                fw.write(' %.2f' % element)
        fw.write('\n')
