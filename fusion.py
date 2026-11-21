"""Fit a held-out-view fusion model and evaluate a frozen model on official cameras."""
import argparse
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiment import Scene,psnr
from layered import Views,Compositor
from surfels import Field,cameras_for,save
from flow import Correspondences


def frame_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a)).cuda().permute(2,0,1)


def rgb_ssim(a,b):
    def g(im):
        return cv2.GaussianBlur(im,(11,11),1.5,borderType=cv2.BORDER_CONSTANT)
    x,y=g(a),g(b)
    return float(np.mean(((2*x*y+.0001)*(2*(g(a*b)-x*y)+.0009))/
                         ((x*x+y*y+.0001)*(g(a*a)+g(b*b)-x*x-y*y+.0009))))


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(15,32,3,padding=1),nn.SiLU(),
                               nn.Conv2d(32,32,3,padding=1),nn.SiLU(),
                               nn.Conv2d(32,24,3,padding=1,dilation=1),nn.SiLU(),
                               nn.Conv2d(24,6,3,padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        delta=self.net(x)
        mean=(x[:,:9].reshape(len(x),3,3,*x.shape[-2:])*delta[:,:3].softmax(1)[:,:,None]).sum(1)
        return mean+.08*delta[:,3:].tanh()


class Renderer:
    def __init__(self,scene,refs,runs):
        self.scene,self.refs=scene,refs
        self.views=Views(scene,refs)
        self.views.build_depth()
        self.compositor=Compositor().cuda().eval()
        self.compositor.load_state_dict(torch.load(runs/"layered"/scene.root.name[:4]/"model.pt",weights_only=True))
        ck=torch.load(runs/"surfels"/scene.root.name[:4]/"best.pt",weights_only=True)
        params=ck["field"]
        self.field=Field([params[k] for k in ("xyz","dc","quat","scale")],ck["extent"])
        self.field.load_state_dict(params)
        self.cameras=cameras_for(scene)
        self.flow=Correspondences()

    @torch.no_grad()
    def render(self,t):
        scene=self.scene
        assert t not in self.refs
        gauss,depth,normals,opacity=self.field(scene,t,self.cameras)
        cubic=torch.empty((3,scene.h,scene.w),device="cuda")
        mix=torch.empty_like(opacity)
        for y in range(0,scene.h,64):
            for x in range(0,scene.w,96):
                a,b=max(0,x-5),max(0,y-5)
                ex,ey=min(scene.w,x+96),min(scene.h,y+64)
                box=(a,b,min(scene.w,ex+5)-a,min(scene.h,ey+5)-b)
                pred,_,alpha=self.compositor(self.views,t,box)
                cubic[:,y:ey,x:ex]=pred[:,y-b:ey-b,x-a:ex-a]
                mix[y:ey,x:ex]=alpha[0,y-b:ey-b,x-a:ex-a]
        field=gauss.permute(2,0,1).clamp(0,1)
        cubic=cubic.clamp(0,1)
        warp,unc=self.flow.render(scene,t,self.refs)
        warp=frame_tensor(warp)
        diff=torch.stack([(a-b).abs().mean(0) for a,b in ((field,cubic),(field,warp),(cubic,warp))])
        feat=torch.cat((field,cubic,warp,opacity[None],mix[None],torch.from_numpy(unc).cuda()[None],diff))
        return feat,depth,normals


def learn(data,ids,steps):
    torch.manual_seed(67)
    rng=random.Random(67)
    net=Fusion().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    for step in range(steps):
        xs,ys=[],[]
        for _ in range(6):
            feat,gt=data[rng.choice(ids)]
            h,w=gt.shape[1:]
            x,y=rng.randrange(w-48),rng.randrange(h-48)
            xs.append(feat[:,y:y+48,x:x+48])
            ys.append(gt[:,y:y+48,x:x+48])
        pred=net(torch.stack(xs))
        err=(pred-torch.stack(ys))[:,:,4:-4,4:-4]
        loss=err.square().mean()+.012*err.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net.eval()


def variants(x,net):
    return dict(surfels=x[:3],two_depth=x[3:6],flow=x[6:9],average=(x[:3]+x[3:6]+x[6:9])/3,
                fusion=net(x[None])[0].clamp(0,1))


def calibrate(scene,args,out):
    dev=sorted(scene.train)[3::19][:12]
    renderer=Renderer(scene,scene.train-set(dev),args.runs)
    data={}
    for t in dev:
        feat,_,_=renderer.render(t)
        data[t]=(feat,frame_tensor(scene.image(t).astype("f")/255))
    fit,check=dev[::2],dev[1::2]
    net=learn(data,fit,600)
    rows=[]
    with torch.no_grad():
        for t in check:
            x,gt=data[t]
            row={k:float(-10*(v-gt).square().mean().log10()) for k,v in variants(x,net).items()}
            rows.append(dict(view=t,**row))
    means={k:float(np.mean([r[k] for r in rows])) for k in row}
    choice=max(means,key=means.get)
    net=learn(data,dev,600)
    save(out/"fusion.pt",dict(net=net.state_dict(),choice=choice))
    (out/"calibration.json").write_text(json.dumps(dict(fit=fit,check=check,means=means,views=rows,choice=choice),indent=2))
    print(scene.root.name[:4],choice,means,flush=True)


@torch.no_grad()
def final(scene,args,out):
    import lpips
    weights=torch.load(out/"fusion.pt",weights_only=True)
    net=Fusion().cuda().eval()
    net.load_state_dict(weights["net"])
    renderer=Renderer(scene,scene.train,args.runs)
    lp=lpips.LPIPS(net="vgg",verbose=False).eval().cuda()
    rows=[]
    for t in sorted(scene.test):
        feat,depth,normal=renderer.render(t)
        value=variants(feat,net)[weights["choice"]]
        pred=value.permute(1,2,0).cpu().numpy()
        gt=scene.image(t).astype("f")/255
        truth=frame_tensor(gt).flip([0])[None]
        rgb=value.flip([0])[None]
        # Original EasyVolCap metric passes [0,1] RGB to LPIPS-VGG directly.
        rows.append(dict(view=t,psnr=psnr(pred,gt),ssim=rgb_ssim(pred,gt),
                         lpips_glint=float(lp(rgb,truth).item()),
                         lpips_standard=float(lp(rgb*2-1,truth*2-1).item())))
        if len(rows)==1+len(scene.test)//2:
            cv2.imwrite(str(out/"test.png"),np.rint(np.concatenate((pred,gt),1)*255).astype("uint8"))
            np.savez_compressed(out/"geometry.npz",view=t,depth=depth.cpu().numpy(),normal=normal.cpu().numpy())
    means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!="view"}
    report=dict(scene=scene.root.name,choice=weights["choice"],n_train=len(scene.train),n_test=len(scene.test),
                resolution=[scene.w,scene.h],mean=means,views=rows)
    (out/"official.json").write_text(json.dumps(report,indent=2))
    print(scene.root.name[:4],"OFFICIAL",means,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("scene",type=Path)
    cli.add_argument("--configs",type=Path,required=True)
    cli.add_argument("--runs",type=Path,default=Path("runs"))
    cli.add_argument("--final",action="store_true")
    args=cli.parse_args()
    cv2.setNumThreads(2)
    scene=Scene(args.scene,args.configs/(args.scene.name+".yaml"))
    out=args.runs/"fusion"/scene.root.name[:4]
    out.mkdir(exist_ok=True,parents=True)
    if args.final:
        final(scene,args,out)
    else:
        calibrate(scene,args,out)
