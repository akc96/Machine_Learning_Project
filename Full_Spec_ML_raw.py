"""
Spectrogram Detection Autoencoder

Autoencoder

@author: Alvin Chen
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

#from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
#import glob
#from os import listdir
#from os.path import isfile, join
import scipy.io

# from tensorflow.keras.datasets import fashion_mnist

#function
#%returns AE MSE to dB representation of spectrogram
def mse2DB(mse):
    return (mse*5)

#normalize assuming min is 0
def thresh(data):
    data[data>1]=1
    return data

"""Create memory allocator and test for GPU"""
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpu), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


"""Data Loading"""
dir_path = "C:/Users/Alvin/Documents/MATLAB/Cam_Dat/"

# directory for full layer files (model building)
dir_test = "C:/Users/Alvin/Documents/MATLAB/Cam_Dat/Testing_training_autoenc/"

sub_files = ["h_nom/","test/","h_test/"]
recon_loc = "recon/" #sub directory with reconstruction data

#MNIST dataset
# (x_train, _), (x_test, _) = fashion_mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

# print (x_train.shape)
# print (x_test.shape)

#%% ### MAT DATA LOADING

mat = scipy.io.loadmat(dir_test+"count_50_raw_spec.mat")

h_dat_t = mat['h_cell'][0].tolist()
h_dat = h_dat_t[0];
for i in h_dat_t[1:]:
    h_dat = np.append(h_dat,i,axis=1);
    
u_dat_t = mat['u_cell'][0].tolist()
u_dat = u_dat_t[0];
for i in u_dat_t[1:]:
    u_dat = np.append(u_dat,i,axis=1);

nom_dat_list = mat['nom_cell'][0].tolist()
nom_dat = nom_dat_list[0];
for i in nom_dat_list[1:]:
    nom_dat = np.append(nom_dat,i,axis=1);

h_test = mat['h_test'][0].tolist()
u_test = mat['u_test'][0].tolist()

#%% Assigning data

global_max = 2 #heuristically modify max score
#Min is about zero, so we will ignore bottom normalization
#threshold high values to 1

img_train = h_dat.transpose()
img_train = thresh(img_train.astype('float32')/global_max)

img_test = nom_dat.transpose()
img_test = thresh(img_test.astype('float32') / global_max)

img_unh = u_dat.transpose()
img_unh = thresh(img_unh.astype('float32') / global_max)

# included in samp_lays
lay_26 = u_test[1].transpose()
lay_26 = thresh(lay_26.astype('float32')/global_max)


#%% display data sizes

print ('Training:')
print (img_train.shape)
print ('Testing:')
print (img_test.shape)
print ('Unhealthy:')
print (img_unh.shape)

#%%
#Construct Autoencoder

latent_dim = 2

print("Latent Dimension:")
print(latent_dim)

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='sigmoid'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(63, activation='sigmoid'),
      layers.Reshape((63, 1))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#%%
#Train autoencoder

history = autoencoder.fit(img_train, img_train, epochs=10,batch_size = 100, shuffle=True, validation_data=(img_test, img_test))

#%% Verify 
encoded_imgs = autoencoder.encoder(img_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

l_spec = 553; #arbitrary shape for plotting
l_spec_full = np.shape(decoded_imgs)[0];

n = 5
k = 0;
test_fig = plt.figure(figsize=(20, 2))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(np.reshape(img_test[l_spec*(i+k):l_spec*(i+k+1)].transpose(),(63,l_spec)))
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(np.reshape(decoded_imgs[l_spec*(i+k):l_spec*(i+k+1)].transpose(),(63,l_spec)))
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

#MSE calculation
MS_test = np.mean(np.square(np.subtract(img_test,np.reshape(decoded_imgs,(l_spec_full,63)))),axis = 1)
MSE_test = mse2DB(np.sqrt(MS_test))

mse_test_fig = plt.figure()
plt.plot(MSE_test)
plt.title("Mean Squared Error, Test Data")
plt.xlabel('Index')
plt.ylabel('MSE')
plt.show()

#%% Unhealthy
encoded_unh = autoencoder.encoder(img_unh).numpy()
decoded_unh = autoencoder.decoder(encoded_unh).numpy()

unh_len_full = np.shape(decoded_unh)[0]
unh_len = 553
k = 0;

# display original
unh_fig = plt.figure()
ax = plt.subplot(2, 1, 1)
plt.imshow(np.reshape(img_unh[unh_len*(k):unh_len*(k+1)].transpose(),(63,unh_len)))
plt.title("original")
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# display reconstruction
ax = plt.subplot(2, 1, 2)
plt.imshow(np.reshape(decoded_unh[unh_len*(k):unh_len*(k+1)].transpose(),(63,unh_len)))
plt.title("reconstructed")
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

#MSE calculation
MS_unh = np.mean(np.square(np.subtract(img_unh,np.reshape(decoded_unh,(unh_len_full,63)))),axis = 1)
MSE_unh = mse2DB(np.sqrt(MS_unh))

mse_unh_fig = plt.figure()
plt.plot(MSE_unh[unh_len*(k):unh_len*(k+1)])
plt.title("Mean Squared Error, Unhealthy Segments")
plt.xlabel('Index')
plt.ylabel('MSE (dB)')
plt.show()

#%% Full layer

len_26 = np.shape(lay_26)[0]
lay_26_t = np.reshape(lay_26.transpose(),(len_26,63,1))

encoded_26 = autoencoder.encoder(lay_26_t).numpy()
decoded_26 = autoencoder.decoder(encoded_26).numpy()

# display original
lay_26_fig = plt.figure()
ax = plt.subplot(2, 1, 1)
plt.imshow(thresh(lay_26.transpose()))
plt.title("original")
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# display reconstruction
ax = plt.subplot(2, 1, 2)
plt.imshow(np.reshape(decoded_26.transpose(),(63,len_26)))
plt.title("reconstructed")
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

MS_26 = np.mean(np.square(np.subtract(lay_26_t,decoded_26)),axis = 1)
MSE_26 = mse2DB(np.sqrt(MS_26))

lay_26_MSE_fig = plt.figure()
plt.plot(MSE_26)
plt.title("Mean Squared Error, Full Test Layer")
plt.xlabel('Index')
plt.ylabel('MSE (dB)')
plt.show()

mse = losses.MeanSquaredError()
MSE_lay = mse2DB(mse(lay_26_t,decoded_26).numpy())
print("Layer MSE (dB)")
print(MSE_lay)

#%%  Plot Frqs

frq_pos = [4,19]
lay_26_recon = np.reshape(np.transpose(decoded_26),np.shape(lay_26))

lay_26_frq_fig = plt.figure()
ax1 = plt.subplot(2,1,1)
plt.ylabel('Mag')
ax2 = plt.subplot(2,1,2)
plt.ylabel('Mag')
plt.xlabel('Index')
frq_legend = []
for i in range(len(frq_pos)):
    ax1.plot(5*(lay_26[frq_pos[i]][:]))
    ax2.plot(5*(lay_26_recon[frq_pos[i]][:]))
    frq_legend.append(str(4*frq_pos[i])+' Hz')

ax1.set_ylim([-0.5,5])
ax2.set_ylim([-0.5,5])
ax1.get_xaxis().set_visible(False)
plt.legend(frq_legend)
ax1.set_title('Original')
ax2.set_title('Reconstructed')

#%% Generate sample recon errors for algorithm

# samp_recon = []

# #loop through sample layers
# for i in range(len(samp_lays)):
#     t_recon_list = []
#     for j in range(len(samp_lays[i])): #for each image in the sample layers
    
#         #generate reconstruction error
#         t_img = samp_lays[i][j]
#         t_encoded = autoencoder.encoder(t_img.transpose()).numpy()
#         t_decoded = autoencoder.decoder(t_encoded).numpy()
#         t_MS = np.mean(np.square(np.subtract(t_img.transpose(),t_decoded.reshape(np.shape(t_img.transpose())))),axis = 1)
#         t_MSE = mse2DB(np.sqrt(t_MS))
        
#         #save data as mat
#         t_fname = samp_fname[i][j]
#         n_fname = recon_loc+sub_files[i]+t_fname[0:len(t_fname)-3]+"mat" #new fname, plus subdirectories
#         scipy.io.savemat(dir_test+n_fname,mdict={'recon':t_MSE})
        
#         t_recon_list.append(t_MSE)
#     samp_recon.append(t_MSE) #save dat to variable for posterity


#recon layers

samp_recon = [];
samp_LS = [];

for t_spec in nom_dat_list:
    t_encoded = autoencoder.encoder(thresh(t_spec.transpose()/global_max)).numpy()
    t_decoded = autoencoder.decoder(t_encoded).numpy()
    t_MS = np.mean(np.square(np.subtract(t_spec.transpose()/global_max,t_decoded.reshape(np.shape(t_spec.transpose())))),axis = 1)
    t_MSE = global_max*(np.sqrt(t_MS))
    
    samp_recon.append(t_MSE)
    samp_LS.append(t_encoded)
    

h_recon = [];
h_LS = [];

for t_spec in h_test:
    t_encoded = autoencoder.encoder(thresh(t_spec.transpose()/global_max)).numpy()
    t_decoded = autoencoder.decoder(t_encoded).numpy()
    t_MS = np.mean(np.square(np.subtract(t_spec.transpose()/global_max,t_decoded.reshape(np.shape(t_spec.transpose())))),axis = 1)
    t_MSE = global_max*(np.sqrt(t_MS))
    
    h_recon.append(t_MSE)
    h_LS.append(t_encoded)


u_recon = [];
u_LS = [];

for t_spec in u_test:
    t_encoded = autoencoder.encoder(thresh(t_spec.transpose()/global_max)).numpy()
    t_decoded = autoencoder.decoder(t_encoded).numpy()
    t_MS = np.mean(np.square(np.subtract(t_spec.transpose()/global_max,t_decoded.reshape(np.shape(t_spec.transpose())))),axis = 1)
    t_MSE = global_max*(np.sqrt(t_MS))
    
    u_recon.append(t_MSE)
    u_LS.append(t_encoded)

#save reconstruction as mat    
scipy.io.savemat(dir_test+"recon.mat",mdict={'nom_recon':samp_recon,'h_recon':h_recon,'u_recon':u_recon})

#save latent space
scipy.io.savemat(dir_test+"LS.mat",mdict={'nom_LS':samp_LS,'h_LS':h_LS,'u_LS':u_LS})


#%% Loss Fig

loss_fig,ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label = 'Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# ax1.set_ylim([0.15, 1.05])
ax.legend(loc='upper right')
ax.set_title('Training and Validation Loss (MSE) vs Epoch')

# fig2,ax2 = plt.subplots()
# ax2.plot(history.history['loss'], label='Training Loss')
# ax2.plot(history.history['val_loss'], label = 'Testing Loss')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Loss')
# ax2.set_ylim([-0.1, 4.8])
# ax2.legend(loc='upper right')
# ax2.set_title('Training and Testing Loss vs Epoch')


#%%

# """Saving Model"""
now=datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

autoencoder.save(dir_path+'/Models/'+date_time+'-autoen')

loss_fig.savefig(dir_path+'Temp Figures/'+date_time+'-mse.png')
# fig2.savefig(dir_path+'Temp Figures/'+date_time+'-loss.png')

test_fig.savefig(dir_path+'Temp Figures/'+date_time+'-test_spec.png')
mse_test_fig.savefig(dir_path+'Temp Figures/'+date_time+'-test_mse.png')

unh_fig.savefig(dir_path+'Temp Figures/'+date_time+'-unh_spec.png')
mse_unh_fig.savefig(dir_path+'Temp Figures/'+date_time+'-unh_mse.png')

lay_26_fig.savefig(dir_path+'Temp Figures/'+date_time+'-lay_26_spec.png')
lay_26_MSE_fig.savefig(dir_path+'Temp Figures/'+date_time+'-lay_26_mse.png')
lay_26_frq_fig.savefig(dir_path+'Temp Figures/'+date_time+'-lay_26_frq.png')

# """Plotting Images"""
# weights = (model.layers[0].weights)[0]
# fig=plt.figure(figsize=(20,10))
# ax=fig.subplots(4,8)
# for i in range(32):
#     img_temp = weights[:,:,:,i].numpy()
#     #normalize image to 0,255
#     img_temp = (255*(img_temp - np.min(img_temp))/np.ptp(img_temp)).astype(int)
#     ax_temp = ax[np.floor(i/8).astype(int),np.mod(i,8)]
#     map_temp = ax_temp.imshow(img_temp)
#     ax_temp.axis('off')
#     ax_temp.set_title('Filter '+str(i+1))
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(map_temp, cax=cbar_ax)
# fig.savefig('Figures/'+date_time+'-Filter.png')
