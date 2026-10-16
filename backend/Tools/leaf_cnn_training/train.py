import numpy as np, tensorflow as tf, json, time
tf.random.set_seed(0)
d=np.load('data96.npz'); Xtr,ytr,Xte,yte=d['Xtr'],d['ytr'].astype('int32'),d['Xte'],d['yte'].astype('int32'); C=len(d['classes'])
L=tf.keras.layers
aug=tf.keras.Sequential([L.RandomFlip('horizontal_and_vertical'),L.RandomRotation(0.15),L.RandomZoom(0.15),
    L.RandomBrightness(0.2,value_range=(0,1)),L.RandomContrast(0.2)])
def block(x,f):
    x=L.Conv2D(f,3,padding='same',use_bias=False)(x); x=L.BatchNormalization()(x); x=L.ReLU()(x)
    return L.MaxPooling2D()(x)
inp=L.Input((96,96,3)); x=block(inp,32); x=block(x,64); x=block(x,128); x=block(x,128); x=block(x,192)
x=L.GlobalAveragePooling2D()(x); x=L.Dropout(0.3)(x); out=L.Dense(C)(x)
model=tf.keras.Model(inp,out)
def prep(x,y,train):
    x=tf.cast(x,tf.float32)/255.0
    if train: x=aug(x,training=True)
    return x,y
tr=tf.data.Dataset.from_tensor_slices((Xtr,ytr)).shuffle(14000,seed=0).batch(64).map(lambda x,y:prep(x,y,True),num_parallel_calls=2).prefetch(2)
te=tf.data.Dataset.from_tensor_slices((Xte,yte)).batch(128).map(lambda x,y:prep(x,y,False))
EP=24
sched=tf.keras.optimizers.schedules.CosineDecay(2e-3, EP*len(tr))
model.compile(tf.keras.optimizers.Adam(sched),tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
t=time.time()
h=model.fit(tr,validation_data=te,epochs=EP,verbose=2)
model.save('leaf_cnn.keras')
print('train time', round(time.time()-t), 's')
json.dump({k:[float(v) for v in vs] for k,vs in h.history.items()}, open('history.json','w'))
