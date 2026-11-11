import numpy as np, cv2
from PIL import Image
im=np.asarray(Image.open('/tmp/claude-0/w/user/stathis.png').convert('RGB')).astype(np.uint8)
H,W,_=im.shape; print(W,H)
OUT='/tmp/claude-0/w/out/'; K=2; FW,FH=1024,1280
def cut(part):
    h,w,_=part.shape
    bg=(part.astype(int).min(2)>225).astype(np.uint8)  # near-white
    # background = white region connected to the border
    n,cc=cv2.connectedComponents(bg)
    border=set(np.unique(np.concatenate([cc[0],cc[-1],cc[:,0],cc[:,-1]])))-{0}
    bgm=np.isin(cc,list(border))
    a=(~bgm).astype(np.uint8)
    n,cc=cv2.connectedComponents(a); sizes=[(cc==i).sum() for i in range(1,n)]; a=(cc==1+int(np.argmax(sizes))).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)); a=cv2.morphologyEx(a,cv2.MORPH_OPEN,k)
    a=cv2.GaussianBlur(cv2.erode(a,np.ones((3,3),np.uint8)).astype(np.float32),(0,0),0.9)
    return np.dstack([part.astype(np.float32),a])
def fit(rgba,box):
    al=rgba[...,3]; ys,xs=np.where(al>0.5); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    crop=rgba[y0:y1+1,x0:x1+1]; bw=x1-x0+1; bh=y1-y0+1
    tw=(box[2]-box[0])*K; th=(box[3]-box[1])*K; f=min(tw/bw,th/bh)
    nw,nh=int(round(bw*f)),int(round(bh*f)); r=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_CUBIC if f>1 else cv2.INTER_AREA)
    out=np.zeros((FH,FW,4),np.float32); ox=int(round((box[0]+box[2])/2*K-nw/2)); oy=int(box[1]*K); out[oy:oy+nh,ox:ox+nw]=r; return out
for name,part in [('stathis-front',im[:,:W//2]),('stathis-back',im[:,W//2:])]:
    o=fit(cut(part),(36,36,476,616))
    if name.endswith('back'): o=o[:,::-1]  # back texture is stored as seen through the front
    px=np.dstack([o[...,:3],o[...,3:]*255]).clip(0,255).astype(np.uint8)
    Image.fromarray(px,'RGBA').save(OUT+name+'.png',optimize=True)
    print(name, o[...,3].max())
