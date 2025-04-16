import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import load_model, save_model
import innvestigate
import pandas as pd
from tqdm import tqdm
from glob import glob
import re
import time
from import_data import load_data
from constant_variables import *

# Read model
dir_input='/home/CBBI/tsaih/Research/' + 'Model_Xnorm13995_new/'
tmpName = '1Layer1D_1LayerDense_conv1024_kernelstride50_dense512_batch254_normX_biasFalse'
model_saved=load_model(dir_input + tmpName + '_model_1'+ '.h5')
model_saved.summary()

dir_output='/home/CBBI/tsaih/Research/' + 'Model_Xnorm13995_new/IG/'
from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

# Read input data
dir_data='/home/CBBI/tsaih/data/'
print('Reading input data......')
input_train, input_labels_train, sample_names_train, gene_names_train = load_data(
    dir_data + "X_data_batch_13995x6813_train_z.txt")
input_val, input_labels_val, sample_names_val, gene_names_val = load_data(
    dir_data + "X_data_batch_13995x6813_val_z.txt")
input_test, input_labels_test, sample_names_test, gene_names_test = load_data(
    dir_data + "X_data_batch_13995x6813_test_z.txt")

x_train = input_train.reshape(input_train.shape[0], 13995, 1)
x_val = input_val.reshape(input_val.shape[0], 13995, 1)
x_test = input_test.reshape(input_test.shape[0], 13995, 1)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
data=np.vstack((x_train, x_val, x_test))
del x_train, x_val, x_test

analyzer_index= innvestigate.create_analyzer("integrated_gradients",  # analysis method identifier #gradient, guided_backprop
                                          model_saved,
                                          neuron_selection_mode="index")  # postprocess='abs' # select the output neuron to analyze

#check which proteins have finished
done_files= glob(dir_output + 'gradient_' + '*.txt')
done_pro=[]
for d in done_files:
    done_pro.append(re.search('gradient_' + '(.*).txt', d).group(1))
print(len(done_pro))

gene_names_pro = proNames()
print(len(gene_names_pro))

start_time=time.time()

#Use ERALPHA (#10 in the list) as an example
for i in gene_names_pro:
    print("{} : {}".format(list.index(gene_names_pro, i) + 1, i))
    if i == "ERALPHA":
        if i in done_pro:
            print('    done')
        else:
            analysis_value_index = np.empty((data.shape[0], data.shape[1]))
            for j in tqdm(range(0, len(data), 1)):
                # print(j)
                analysis_index = analyzer_index.analyze(data[j:j + 1], neuron_selection=list.index(gene_names_pro, i))
                value_index = analysis_index.reshape(1, 13995)
                analysis_value_index[j] = value_index
                pass
            np.savetxt(dir_output +
                    'gradient_' + i + '.txt', analysis_value_index.transpose(), delimiter='\t', fmt='%.10f')
        
print('END')

end_time=time.time()
elapsed_time = end_time - start_time

# Convert elapsed time to hours if it's greater than 3600 seconds (1 hour)
if elapsed_time >= 3600:
    elapsed_hours = elapsed_time / 3600
    print("Time taken:", elapsed_hours, "hours")
elif elapsed_time >= 60:
    elapsed_minutes = elapsed_time / 60
    print("Time taken:", elapsed_minutes, "minutes")
else:
    print("Time taken:", elapsed_time, "seconds")
