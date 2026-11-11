import sys, numpy as np, onnxruntime as ort
from PIL import Image
sess=ort.InferenceSession('/tmp/claude-0/w/model/seg.onnx')
inp=sess.get_inputs()[0].name
for f in sys.argv[1:]:
    im=Image.open(f).convert('RGB'); W,H=im.size
    x=np.asarray(im.resize((512,512),Image.BILINEAR),dtype=np.float32)/255.
    x=(x-[0.485,0.456,0.406])/[0.229,0.224,0.225]; x=x.transpose(2,0,1)[None].astype(np.float32)
    lg=sess.run(None,{inp:x})[0][0]  # C,h,w
    # upsample logits
    C=lg.shape[0]; up=np.stack([np.asarray(Image.fromarray(lg[c]).resize((W,H),Image.BILINEAR)) for c in range(C)])
    lab=up.argmax(0).astype(np.uint8)
    np.save(f.replace('.jpg','.lab.npy'),lab)
    print(f, {int(k):int(v) for k,v in zip(*np.unique(lab,return_counts=True))})
    pal=np.random.RandomState(1).randint(0,255,(18,3)).astype(np.uint8); pal[0]=0; pal[4]=[255,255,255]
    Image.fromarray(pal[lab]).resize((W//4,H//4)).save(f.replace('.jpg','.lab.png'))
