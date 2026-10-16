import numpy as np, tensorflow as tf, sys, json
sys.path.insert(0,'../..')   # the backend folder
m=tf.keras.models.load_model('leaf_cnn.keras'); d=np.load('data96.npz')
convs=[l for l in m.layers if isinstance(l,tf.keras.layers.Conv2D)]
bns=[l for l in m.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
dense=[l for l in m.layers if isinstance(l,tf.keras.layers.Dense)][0]
out={}
for i,(c,bn) in enumerate(zip(convs,bns)):
    w=c.get_weights()[0]; g,b,mu,var=bn.get_weights(); s=g/np.sqrt(var+bn.epsilon)
    out[f'conv{i}_w']=(w*s).astype(np.float32); out[f'conv{i}_b']=(b-mu*s).astype(np.float32)
W,B=dense.get_weights(); out['dense_w']=W.astype(np.float32); out['dense_b']=B.astype(np.float32)
out['classes']=d['classes']
Xte,yte=d['Xte'],d['yte']
pk=m.predict(Xte.astype('float32')/255.0,verbose=0).argmax(1)
acc=float((pk==yte).mean()); out['test_accuracy']=np.float32(acc)
np.savez_compressed('leaf_cnn.npz',**out)
from leaf_model import LeafClassifier
clf=LeafClassifier('leaf_cnn.npz')
import time; t=time.time()
pn=np.array([clf.classes.index(clf.predict_pixels(x,1)[0]['class']) for x in Xte[:300]])
print('keras test acc',round(acc,4),'numpy agrees with keras on',(pn==pk[:300]).mean(),'time/img',round((time.time()-t)/300,4))
# per-class accuracy + top-3
P=tf.nn.softmax(m.predict(Xte.astype('float32')/255.0,verbose=0)).numpy()
top3=np.mean([yte[i] in np.argsort(-P[i])[:3] for i in range(len(yte))])
per={str(d['classes'][c]):round(float((pk[yte==c]==c).mean()),3) for c in range(len(d['classes']))}
json.dump({'test_accuracy':acc,'top3_accuracy':float(top3),'test_images':int(len(yte)),'train_images':int(len(d['ytr'])),'per_class':per},open('report.json','w'),indent=1)
print('top3',round(top3,4)); print(sorted(per.items(),key=lambda kv:kv[1])[:6])
