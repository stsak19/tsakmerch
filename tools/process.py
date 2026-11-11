import numpy as np, cv2, json, sys
from PIL import Image
from scipy import ndimage as ndi
SRC='/tmp/claude-0/w/src/'; OUT='/tmp/claude-0/w/out/'
import os; os.makedirs(OUT,exist_ok=True)
FW,FH=1024,1280; K=2  # frame = 512x640 garment frame at 2x
def cutout(name, fg, holes=(), work=1100):
    im=np.asarray(Image.open(SRC+name+'.jpg').convert('RGB')); lab=np.load(SRC+name+'.lab.npy')
    H,W=lab.shape; sc=work/max(H,W)
    ims=cv2.resize(im,(int(W*sc),int(H*sc)),interpolation=cv2.INTER_AREA)
    labs=cv2.resize(lab,(ims.shape[1],ims.shape[0]),interpolation=cv2.INTER_NEAREST)
    m=np.isin(labs,fg).astype(np.uint8)
    # keep largest component
    n,cc=cv2.connectedComponents(m); 
    if n>2: sizes=[(cc==i).sum() for i in range(1,n)]; m=(cc==1+int(np.argmax(sizes))).astype(np.uint8)
    hole=np.isin(labs,holes)
    gm=np.full(m.shape,cv2.GC_BGD,np.uint8)
    dil=cv2.dilate(m,np.ones((25,25),np.uint8)); ero=cv2.erode(m,np.ones((15,15),np.uint8))
    gm[dil>0]=cv2.GC_PR_BGD; gm[m>0]=cv2.GC_PR_FGD; gm[ero>0]=cv2.GC_FGD
    gm[hole]=cv2.GC_BGD
    # skin/pants are certain background
    gm[np.isin(labs,[6,11,12,13,14,15,2])]=cv2.GC_BGD
    bgd=np.zeros((1,65)); fgd=np.zeros((1,65))
    cv2.grabCut(cv2.cvtColor(ims,cv2.COLOR_RGB2BGR),gm,None,bgd,fgd,6,cv2.GC_INIT_WITH_MASK)
    a=np.isin(gm,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
    a=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,cc=cv2.connectedComponents(a); sizes=[(cc==i).sum() for i in range(1,n)]; a=(cc==1+int(np.argmax(sizes))).astype(np.uint8)
    filled=ndi.binary_fill_holes(a).astype(np.uint8)
    holefill=(filled>0)&(a==0)
    return ims, a, holefill
def shade_of(ims,a):
    g=cv2.cvtColor(ims,cv2.COLOR_RGB2GRAY).astype(np.float32)
    med=np.median(g[a>0]); return g/med
def fit(rgba, box, top_align=True):
    # rgba float HxWx4 ; box in 512-frame coords x0,y0,x1,y1
    al=rgba[...,3]; ys,xs=np.where(al>0.5); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    crop=rgba[y0:y1+1,x0:x1+1]; bw=x1-x0+1; bh=y1-y0+1
    tw=(box[2]-box[0])*K; th=(box[3]-box[1])*K; f=min(tw/bw,th/bh)
    nw,nh=int(round(bw*f)),int(round(bh*f))
    r=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA if f<1 else cv2.INTER_CUBIC)
    out=np.zeros((FH,FW,4),np.float32); cx=(box[0]+box[2])/2*K; ox=int(round(cx-nw/2)); oy=int(box[1]*K)
    out[oy:oy+nh,ox:ox+nw]=r; return out
def save(rgba,name,mode):
    o=rgba.copy(); o[...,3]=np.clip(o[...,3],0,1)
    if mode=='shade':
        v=np.clip(o[...,0]*200,0,255); Image.fromarray(np.dstack([v,o[...,3]*255]).astype(np.uint8),'LA').save(OUT+name+'.png',optimize=True); return
    else: px=np.dstack([o[...,:3],o[...,3:]*255])
    Image.fromarray(px.astype(np.uint8),'RGBA').save(OUT+name+'.png',optimize=True)
def soft(a):
    a=a.astype(np.uint8); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    a=cv2.morphologyEx(a,cv2.MORPH_OPEN,k); a=cv2.morphologyEx(a,cv2.MORPH_CLOSE,k)
    a=cv2.GaussianBlur(a.astype(np.float32),(0,0),2.0); a=(a>0.6).astype(np.float32)   # smooth contour, ~2px inset
    return cv2.GaussianBlur(a,(0,0),0.8)
meta={}
def rec(name,rgba):
    al=rgba[...,3]>0.5; ys,xs=np.where(al); W=xs.max()-xs.min()
    cx=(xs.max()+xs.min())//2
    def topat(x): c=np.where(al[:,x])[0]; return int(c.min()) if len(c) else -1
    meta[name]=dict(top=int(ys.min())/K, bottom=int(ys.max())/K, left=int(xs.min())/K, right=int(xs.max())/K,
        centerTop=topat(cx)/K, shoulderL=topat(int(cx-W*0.18))/K, shoulderR=topat(int(cx+W*0.18))/K)

# ---- tee (flat lay)
ims,a,_=cutout('12039633',[4,7])
s=shade_of(ims,a); rg=np.dstack([s,s,s,soft(a)])
tee=fit(rg,(36,36,476,616)); save(tee,'tee-front','shade'); rec('tee-front',tee)
# back: mirror, close the front neck dip so the back neckline is high
teeb=tee[:,::-1].copy(); save(teeb,'tee-back','shade'); rec('tee-back',teeb)
np.save(OUT+'tee.npy',tee)
# ---- hoodie front (worn, hood up) : face becomes dark hood interior, rim rebuilt around it
ims,a,hf=cutout('8217400',[4],holes=[11,2])
lab=np.load(SRC+'8217400.lab.npy'); labs=cv2.resize(lab,(ims.shape[1],ims.shape[0]),interpolation=cv2.INTER_NEAREST)
head=np.isin(labs,[11,2]).astype(np.uint8)
n,cc=cv2.connectedComponents(head); sizes=[(cc==i).sum() for i in range(1,n)]; head=(cc==1+int(np.argmax(sizes))).astype(np.uint8)
head=ndi.binary_fill_holes(head|hf).astype(np.uint8)
# the head stops where the garment starts: chin line
ys,xs=np.where(head); hw=xs.max()-xs.min()
rim=cv2.dilate(head,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(hw*.22)|1,int(hw*.22)|1)))
rim[ys.max():]=0
s=shade_of(ims,a)
ring=(rim>0)&(head==0)&(a==0)
s=np.where(ring,np.clip(s,0.86,1.05),s)
dist=cv2.distanceTransform(head,cv2.DIST_L2,5); dn=dist/(dist.max()+1e-6)
t=np.clip((np.arange(s.shape[0])[:,None]-ys.min())/max(1,ys.max()-ys.min()),0,1)
noise=cv2.GaussianBlur(np.random.RandomState(3).rand(*s.shape).astype(np.float32),(0,0),4)*0.05
interior=0.36-0.17*np.clip(dn,0,1)**.5+0.06*t+noise
s=np.where(head>0,interior,s)
# soft inner shadow of the rim onto the interior
al=soft(np.maximum(np.maximum(a,head),(rim>0).astype(np.uint8)))
hood=fit(np.dstack([s,s,s,al]),(38,6,474,626)); save(hood,'hoodie-front','shade'); rec('hoodie-front',hood)
# ---- hoodie back
ims,a,_=cutout('8217415',[4])
s=shade_of(ims,a); hb=fit(np.dstack([s,s,s,soft(a)]),(38,6,474,626))
hb=hb[:,::-1].copy()  # back texture is drawn mirrored on the back face; store as-seen-from-front orientation
save(hb,'hoodie-back','shade'); rec('hoodie-back',hb)
# ---- black tee (real colour, hanging)
im0=np.asarray(Image.open(SRC+'8532616.jpg').convert('RGB'))
ims,a,_=cutout('8532616',[4,7])
rgb=ims.astype(np.float32)
bt=fit(np.dstack([rgb,soft(a)]),(36,36,476,616)); save(bt,'tee-black-front','raw'); rec('tee-black-front',bt)
json.dump(meta,open(OUT+'meta.json','w'),indent=1); print(json.dumps(meta,indent=1))
