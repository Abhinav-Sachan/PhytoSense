import os, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
ROOT='../pv/raw/color'; S=96
classes=sorted(os.listdir(ROOT))
def load(path):
    im=Image.open(path).convert('RGB').resize((S,S), Image.BILINEAR)
    return np.asarray(im,dtype=np.uint8)
items=[(os.path.join(ROOT,c,f),i) for i,c in enumerate(classes) for f in sorted(os.listdir(os.path.join(ROOT,c)))]
with ProcessPoolExecutor(2) as ex: X=np.stack(list(ex.map(load,[p for p,_ in items],chunksize=64)))
y=np.array([i for _,i in items],dtype=np.int16)
rng=np.random.default_rng(0); te=np.zeros(len(y),bool)
for c in range(len(classes)):
    idx=np.where(y==c)[0]; rng.shuffle(idx); te[idx[:max(20,int(len(idx)*0.15))]]=True
np.savez('data96.npz',Xtr=X[~te],ytr=y[~te],Xte=X[te],yte=y[te],classes=np.array(classes))
print(X.shape, (~te).sum(), te.sum())
