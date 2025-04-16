from __future__ import print_function, division
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from constant_variables import *
import numpy as np
from import_data import load_data
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Lambda, Input, Dense, Input, Flatten, Multiply, Reshape, concatenate
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Conv2DTranspose, \
    BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.callbacks import Callback
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
#K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=100, inter_op_parallelism_threads=100)))
tf.config.threading.set_intra_op_parallelism_threads(100)
tf.config.threading.set_inter_op_parallelism_threads(100)

# Sampling function as a layer
def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
    return random_sample

# VAE loss function
class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight

    def on_epoch_end(self, epoch, logs={}):
        if epoch > klstart:
            new_weight = min(K.get_value(self.weight) + (1. / kl_annealtime), 1.)
            K.set_value(self.weight, new_weight)
        print("Current KL Weight is " + str(K.get_value(self.weight)))

def vae_reconstruction_loss(y_true, y_predict):
    experimental_run_tf_function = False
    reconstruction_loss_factor = 13995
    reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_predict))
    return 0.5 * reconstruction_loss_factor * reconstruction_loss

def vae_kl_loss(encoder_mu, encoder_log_variance):
    experimental_run_tf_function = False
    kl_loss = -0.5 * K.sum(1 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance))
    return 0.5 * kl_loss

def loss(encoder_log_variance, encoder_mu, weight=None):
    experimental_run_tf_function = False

    def vae_loss(true, pred):
        reconstruction_loss = mse(K.flatten(true), K.flatten(pred))
        reconstruction_loss *= 13995
        kl_loss = -0.5 * K.sum(1 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance))
        # kl_loss=vae_kl_loss
        if weight is None:
            return K.mean(reconstruction_loss + kl_loss)
        if weight is not None:
            return (0.5 * reconstruction_loss) + (weight * 0.5 * kl_loss)

    return vae_loss

# Plot the scatterplots
def plot_figure(df, tsne_results, tumor_name, filename=None):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    data=df, s=20).set(title="Scatter Plot")

    means = np.vstack([tsne_results[tumor_name == i].mean(axis=0) for i in (df.y.tolist())])

    sns.scatterplot(means[:, 0], means[:, 1], hue=df.y.tolist(), s=20, ec='black', legend=False)

    for j, i in enumerate(np.unique(df.y.tolist())):
        plt.annotate(i, df.loc[df['y'] == i, ['comp-1', 'comp-2']].mean(),
                     horizontalalignment='center',
                     verticalalignment='top',
                     size=12, weight='bold')
    plt.savefig(filename)

dir_input='/home/CBBI/tsaih/data/'
dir_output = '/home/CBBI/tsaih/Research/diffMLmethods_comparison/CrossValidation_Xnorm_Z01/VAE_hyperas/'

from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

klstart = 10
kl_annealtime = 20

dir_input = "/home/CBBI/tsaih/data/"
X_train, _, _, _ = load_data(dir_input + "X_data_batch_13995x6813_train_z.txt")
print(X_train.shape)
X_test, _, _, _ = load_data(dir_input + "X_data_batch_13995x6813_test_z.txt")
print(X_test.shape)
X_val, _, _, _ = load_data(dir_input + "X_data_batch_13995x6813_val_z.txt")
print(X_val.shape)

weight = K.variable(0.)
n_epochs = 100
latent_space_dim = 150
lr=0.0005
batch_size=256
L1=1024
numLayer='one'

# Initialize Encoder Model
x = Input(shape=(13995,), name="encoder_input")
encoder_layer1 = Dense(L1, name='encoder_layer_1')(x)
encoder_norm_layer1 = BatchNormalization(name="encoder_norm_1")(encoder_layer1)
encoder_activ_layer1 = LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)
encoder_mu = Dense(units=latent_space_dim, name="encoder_mu")(encoder_activ_layer1)
encoder_log_variance = Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_activ_layer1)
z = Lambda(sampling, name='z_sample')([encoder_mu, encoder_log_variance])
encoder_model = Model(x, [encoder_mu, encoder_log_variance, z], name="encoder_model")

# Initialize Decoder Model
decoder_input = Input(shape=(latent_space_dim,), name="decoder_input")
decoder_dense_layer1 = Dense(units=L1, name="decoder_dense_1")(decoder_input)
decoder_norm_layer1 = BatchNormalization(name="decoder_norm_1")(decoder_dense_layer1)
decoder_activ_layer1 = LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)
decoder_dense_layer_out = Dense(units=13995, name="decoder_dense_out")(decoder_activ_layer1)
decoder_norm_layer_out = BatchNormalization(name="decoder_norm_out")(decoder_dense_layer_out)
decoder_output = LeakyReLU(name="decoder_leakyrelu_out")(decoder_norm_layer_out)

decoder_model = Model(decoder_input, decoder_output, name="decoder_model")

# Define VAE
vae_encoder_output = encoder_model(x)
vae_decoder_output = decoder_model(encoder_model(x)[2])
vae = Model(x, vae_decoder_output, name="VAE")

# For training enable this
if 1 == 1:
    # Training
    opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-3)
    vae.compile(optimizer=opt, loss=loss(encoder_log_variance, encoder_mu, weight), metrics=[vae_reconstruction_loss, vae_kl_loss])
    history2 = vae.fit(X_train, X_train, epochs=n_epochs, validation_data=(X_val, X_val), batch_size=batch_size, shuffle=False) #, verbose=0)

    validation_loss = np.amin(history2.history['val_loss'])
    cost = vae.evaluate(X_test, X_test, verbose=0)
    params = vae.count_params()

    # save model
    encoder_model.save([dir_output+'a_encoder.h5'][0])
    decoder_model.save([dir_output+'a_decoder.h5'][0])
    vae.save([dir_output+'a_vae.h5'][0])

    # Plot the training and validation loss
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title("Model Loss (MSE Reconstruction Loss)")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Val'])
    plt.savefig([dir_output+'Figure_ModelLoss_MSE.png'][0])
    plt.show()
    plt.close()

    # Plot RL and KL loss during training
    plt.plot(history2.history['vae_reconstruction_loss'])
    plt.title('Training Reconstruction Loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig([dir_output+'Figure_Training_RL.png'][0])
    plt.show()
    plt.close()

    plt.plot(history2.history['vae_kl_loss'])
    plt.title('Training KL Loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig([dir_output+'Figure_Training_KL.png'][0])
    plt.show()
    plt.close()

# To load an existing model
if 1 == 0:
    encoder_model.load_weights([dir_output + 'a_encoder.h5'][0])
    decoder_model.load_weights([dir_output + 'a_decoder.h5'][0])
    vae.load_weights([dir_output + 'a_vae.h5'][0])

#train data
X_tcga_mu, X_tcga_std, X_tcga_dsample = encoder_model.predict(X_train)
if latent_space_dim > 2:
    X_tcga_mu = TSNE(n_components=2, perplexity=40).fit_transform(X_tcga_mu)

df = pd.DataFrame()
df["sampleID"] = sampleTrain()
df["comp-1"] = X_tcga_mu[:, 0]
df["comp-2"] = X_tcga_mu[:, 1]
df=pd.merge(df, samples2tumortypes()[['sample', 'Disease']], left_on='sampleID', right_on='sample')

Y_train, data_labels_pro, sample_names_pro, gene_names_pro = load_data(dir_input+"Y_data_187x6813_train" + ".txt")
index = gene_names_pro.index('ERALPHA')
df['target']=Y_train[:,index]

sns.scatterplot(x='comp-1', y='comp-2', data=df, hue='Disease')
sns.scatterplot(x='comp-1', y='comp-2', data=df, hue='target')

plt.tight_layout()
plt.savefig(dir_output + 'TSNE_Encoder_train' + '.png', format='png', dpi=600)
plt.savefig(dir_output + 'TSNE_Encoder_train' +  '.pdf', format='pdf', dpi=600)
plt.close()

df.head

# Original plot
sns.scatterplot(x='comp-1', y='comp-2', data=df, hue='Disease', s=20, legend=False)

# Calculate and plot group means
means_df = df.groupby('Disease')[['comp-1', 'comp-2']].mean().reset_index()
sns.scatterplot(x='comp-1', y='comp-2', data=means_df, hue='Disease', 
                s=80, edgecolor='black', legend=False, marker='X')

# Annotate the means
for _, row in means_df.iterrows():
    plt.annotate(row['Disease'], (row['comp-1'], row['comp-2']),
                 horizontalalignment='center', verticalalignment='top',
                 fontsize=10)

sns.scatterplot(x='comp-1', y='comp-2', data=df, hue='target')

#val data
X_val_mu, X_val_std, X_val_dsample = encoder_model.predict(X_val)
if latent_space_dim > 2:
    X_val_mu = TSNE(n_components=2, perplexity=40).fit_transform(X_val_mu)
df = pd.DataFrame()
df["sampleID"] = sampleVal()
df["comp-1"] = X_val_mu[:, 0]
df["comp-2"] = X_val_mu[:, 1]
df=pd.merge(df, samples2tumortypes()[['sample', 'Disease']], left_on='sampleID', right_on='sample')

sns.scatterplot(x='comp-1', y='comp-2', data=df, hue='Disease')

plt.tight_layout()
plt.savefig(dir_output + 'TSNE_Encoder_val' + '.png', format='png', dpi=600)
plt.savefig(dir_output + 'TSNE_Encoder_val' +  '.pdf', format='pdf', dpi=600)
plt.close()

# Test data
X_test_mu, X_test_std, X_test_dsample = encoder_model.predict(X_test)
if latent_space_dim > 2:
    X_test_mu = TSNE(n_components=2, perplexity=40).fit_transform(X_test_mu)
df = pd.DataFrame()
df["sampleID"] = sampleTest()
df["comp-1"] = X_test_mu[:, 0]
df["comp-2"] = X_test_mu[:, 1]
df=pd.merge(df, samples2tumortypes()[['sample', 'Disease']], left_on='sampleID', right_on='sample')

sns.scatterplot(x='comp-1', y='comp-2', data=df, hue='Disease', legend = False)

plt.tight_layout()
plt.show()
# plt.savefig(dir_output + 'TSNE_Encoder_test' + '.png', format='png', dpi=600)
# plt.savefig(dir_output + 'TSNE_Encoder_test' +  '.pdf', format='pdf', dpi=600)
# plt.close()

# Decoder Visualization
X_test_decoded = decoder_model.predict(X_test_dsample)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results_decoded = tsne.fit_transform(X_test_decoded)
sns.scatterplot(tsne_results_decoded[:, 0], tsne_results_decoded[:, 1])
plt.show()

X_tcga_decoded = decoder_model.predict(X_tcga_dsample)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_tcga_decoded = tsne.fit_transform(X_tcga_decoded)
sns.scatterplot(tsne_tcga_decoded[:, 0], tsne_tcga_decoded[:, 1])
plt.show()

# Save sampled latent vector
np.savetxt(dir_output + 'VAE_sampling_train.txt', X_tcga_dsample.T , delimiter='\t', fmt='%.10f')
np.savetxt(dir_output + 'VAE_sampling_val.txt', X_val_dsample.T , delimiter='\t', fmt='%.10f')
np.savetxt(dir_output + 'VAE_sampling_test.txt', X_test_dsample.T , delimiter='\t', fmt='%.10f')


# # Cross Validation
from keras.callbacks import EarlyStopping
from keras import models
from scipy.stats import pearsonr

def VAEmodel(x_train, x_val, x_test, y_train, y_val, y_test):
    input_dim = x_train.shape[1]
    batch_size = 256
    dense1 =  2048
    act_d1 = 'relu'
    act_out = 'linear'
    n_epoch = 100

    K.clear_session()
    model = models.Sequential()
    model.add(Dense(dense1, input_dim=input_dim, activation=act_d1))
    model.add(Dense(y_train.shape[1], activation=act_out))
    model.compile(optimizer='adam', loss='mse')
    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
    history_2 = model.fit(x_train, y_train,
                          validation_data=(x_val, y_val),
                          batch_size=batch_size,
                          epochs=n_epoch,
                          shuffle=False,
                          callbacks=[history],
                          verbose=0)
    cost = model.evaluate(x_test, y_test, verbose=0)
    params = model.count_params()

    mse_train = history_2.history['loss'][history.stopped_epoch]
    mse_val = history_2.history['val_loss'][history.stopped_epoch]
    mse_test = cost
    epochs = history.stopped_epoch

    #print('Predicting....')
    pred_train = pd.DataFrame(model.predict(x_train, batch_size=batch_size, verbose=0))
    pred_val = pd.DataFrame(model.predict(x_val, batch_size=batch_size, verbose=0))
    pred_test = pd.DataFrame(model.predict(x_test, batch_size=batch_size, verbose=0))
    return mse_train, mse_val, mse_test, pred_train, pred_val, pred_test, epochs


def MSEresult(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    mse_train= mean_squared_error(y_train, pred_train)
    mse_val = mean_squared_error(y_val, pred_val)
    mse_test = mean_squared_error(y_test, pred_test)
    return mse_train, mse_val, mse_test

def Pearsonresult(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    pearsonr_train = []
    pearsonp_train = []
    pearsonr_val = []
    pearsonp_val = []
    pearsonr_test = []
    pearsonp_test = []

    for j in range(y_train.shape[1]):
        pearsonr_train.append(pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[0])
        pearsonp_train.append(pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[1])
        pearsonr_val.append(pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[0])
        pearsonp_val.append(pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[1])
        pearsonr_test.append(pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[0])
        pearsonp_test.append(pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[1])

    CorMatrix = pd.DataFrame({
        'rppa': y_train.columns,
        'pearsonr_train': pearsonr_train,
        'pearsonp_train': pearsonp_train,
        'pearsonr_val': pearsonr_val,
        'pearsonp_val': pearsonp_val,
        'pearsonr_test': pearsonr_test,
        'pearsonp_test': pearsonp_test
    })
    return CorMatrix

# Read VAE input
ori_exp1 = pd.read_csv(dir_output + 'VAE_sampling_train.txt', sep="\t", header=None)
ori_exp2 = pd.read_csv(dir_output + 'VAE_sampling_val.txt', sep="\t", header=None)
ori_exp3 = pd.read_csv(dir_output + 'VAE_sampling_test.txt', sep="\t", header=None)
ori_exp=pd.merge(pd.merge(ori_exp1, ori_exp2, left_index=True, right_index=True), ori_exp3, left_index=True, right_index=True)
del ori_exp1, ori_exp2, ori_exp3

ori_exp.columns=sampleTrain()+sampleVal()+sampleTest()

ori_pro1=pd.read_csv(dir_input + 'Y_data_187x6813_train.txt', sep="\t", index_col=[0])
ori_pro2=pd.read_csv(dir_input + 'Y_data_187x6813_val.txt', sep="\t", index_col=[0])
ori_pro3=pd.read_csv(dir_input + 'Y_data_187x6813_test.txt', sep="\t", index_col=[0])

ori_pro=pd.merge(pd.merge(ori_pro1, ori_pro2, left_index=True, right_index=True), ori_pro3, left_index=True, right_index=True)
del ori_pro1, ori_pro2, ori_pro3

ori_pro.index=ori_pro.index.str.upper()
ori_exp_T=ori_exp.T
ori_pro_T=ori_pro.T

pairing=CorrRNAnPro()

index_train=pd.read_csv(dir_input+'RandomIndex_train_10xCV.txt', sep='\t')
index_val=pd.read_csv(dir_input+'RandomIndex_val_10xCV.txt', sep='\t')
index_test=pd.read_csv(dir_input+'RandomIndex_test_10xCV.txt', sep='\t')

# Define per-fold score containers
tableVAE = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])

# Run one fold as example
for i in range(1):
    print('Round' + str(i+1))

    # input_data
    x_train = ori_exp_T.iloc[index_train.iloc[:, i]]
    x_val = ori_exp_T.iloc[index_val.iloc[:, i]]
    x_test = ori_exp_T.iloc[index_test.iloc[:, i]]
    # output_data
    y_train = ori_pro_T.iloc[index_train.iloc[:, i]]
    y_val = ori_pro_T.iloc[index_val.iloc[:, i]]
    y_test = ori_pro_T.iloc[index_test.iloc[:, i]]

    # VAE
    print('.......VAE')
    resultVAE= VAEmodel(x_train, x_val, x_test, y_train, y_val, y_test)
    pearsonVAE=Pearsonresult(y_train, y_val, y_test, resultVAE[3], resultVAE[4], resultVAE[5])

    tableVAE.loc[i, 'mse_train']=resultVAE[0]
    tableVAE.loc[i, 'mse_val'] = resultVAE[1]
    tableVAE.loc[i, 'mse_test'] = resultVAE[2]
    tableVAE.loc[i, 'cor_train']= pearsonVAE['pearsonr_train'].mean()
    tableVAE.loc[i, 'cor_val'] = pearsonVAE['pearsonr_val'].mean()
    tableVAE.loc[i, 'cor_test'] = pearsonVAE['pearsonr_test'].mean()

tableVAE.to_csv(dir_output + 'Table_result_10foldCrossValidation_' + str(1) + '_VAE.txt', sep='\t')