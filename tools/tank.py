import numpy as np, cv2
from PIL import Image
OUT='/tmp/claude-0/w/out/'; K=2
def q(p0,c,p1,n=30): t=np.linspace(0,1,n)[:,None]; return (1-t)**2*np.array(p0)+2*(1-t)*t*np.array(c)+t**2*np.array(p1)
pts=[]
def M(*p): pts.append(np.array([p]))
cur=[176,26]
def L(x,y):
    global cur; pts.append(np.array([[x,y]])); cur=[x,y]
def Q(cx,cy,x,y):
    global cur; pts.append(q(cur,[cx,cy],[x,y])[1:]); cur=[x,y]
def build(depth):
    global pts,cur
    pts=[]; cur=[176,26]
    pts.append(np.array([[176,26]])); L(214,26); Q(256,depth,298,26); L(336,26)
    Q(338,172,374,236); Q(366,420,376,612); Q(256,624,136,612); Q(146,420,138,236); Q(174,172,176,26)
    P=np.vstack(pts).astype(np.float64)
    P[:,0]=131+(P[:,0]-136)*sx; P[:,1]=36+(P[:,1]-26)*sy
    return P*K
sx=261/240; sy=418/590

for side in ['front','back']:
    im=np.asarray(Image.open(OUT+f'tee-{side}.png').convert('RGBA')).astype(np.float32)
    m=np.zeros(im.shape[:2],np.uint8)
    PP=build(190 if side=='front' else 70)
    if side=='back':  # higher neck at the back
        pass
    cv2.fillPoly(m,[PP.round().astype(np.int32)],1,lineType=cv2.LINE_AA)
    m=cv2.GaussianBlur(m.astype(np.float32),(0,0),1.0)
    a=im[...,3]/255*m
    s=im[...,0].copy()
    # rib binding along neck & armholes (upper part only)
    mb=(m>0.5).astype(np.uint8); er=cv2.erode(mb,np.ones((17,17),np.uint8))
    band=(mb>0)&(er==0); band[int(250*K):]=False
    s[band]*=0.94
    line=(cv2.dilate(er,np.ones((3,3),np.uint8))>0)&(er==0)&(np.arange(s.shape[0])[:,None]<250*K)
    s[line]*=0.86
    # vertical ribs, subtle, everywhere (ribbed tank fabric)
    bandimg=np.zeros_like(s); bandimg[cv2.GaussianBlur(band.astype(np.float32),(0,0),1.2)>0.5]=255
    bandimg=cv2.GaussianBlur(bandimg,(0,0),1.0)
    out=np.dstack([s,bandimg,s,a*255]).clip(0,255).astype(np.uint8)
    Image.fromarray(out,'RGBA').save(OUT+f'tank-{side}.png',optimize=True)
print('ok')
