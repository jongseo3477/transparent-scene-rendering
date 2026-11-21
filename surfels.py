"""Scene-specific 2D Gaussian reconstruction using CUDA rasterization only."""
import argparse
import json
import math
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from gsplat.rendering import rasterization_2dgs

from experiment import Scene, psnr, ssim
from layered import Views, Compositor


def points_from_pixels(scene, t, d, pix, rgb, extent):
    device = d.device
    def gpu(a):
        return torch.as_tensor(a,device=device)
    r, k, center = gpu(scene.R[t]),gpu(scene.K[t]),gpu(scene.C[t])
    yy,xx = pix//scene.w, pix%scene.w
    depth = (1/d[yy,xx].clamp_min(.1/extent)).clamp(.015*extent,10*extent)
    rays = torch.stack((xx,yy,torch.ones_like(xx)),1).float() @ torch.linalg.inv(k).T
    xyz = (rays*depth[:,None]) @ r + center
    color = rgb[:,yy,xx].T.clamp(.005,.995)
    quat_xyzw = Rotation.from_matrix(scene.R[t].T).as_quat()
    quat = gpu(quat_xyzw[[3,0,1,2]].astype("f")).expand(len(pix),4).clone()
    scale = (depth / k[0,0] * 2.1).log()[:,None].expand(-1,2).clone()
    return xyz,color,quat,scale


class Field(nn.Module):
    def __init__(self, args, extent):
        super().__init__()
        xyz,color,quat,scale = args
        self.xyz = nn.Parameter(xyz)
        self.dc = nn.Parameter(color)
        self.quat = nn.Parameter(quat)
        self.scale = nn.Parameter(scale)
        self.alpha = nn.Parameter(torch.full_like(color[:,0],-.8))
        self.sh = nn.Parameter(xyz.new_zeros(len(xyz),8,3))
        self.background = nn.Parameter(xyz.new_tensor([.3,.3,.3]))
        self.extent = extent

    def forward(self, scene, t, cameras):
        view,k,c = cameras[t]
        x,y,z = F.normalize(self.xyz.detach()-c,dim=1).unbind(1)
        basis = torch.stack((x,y,z,x*y,y*z,z*x,x*x-y*y,3*z*z-1),1)
        color = self.dc + (self.sh*basis[:,:,None]).sum(1)
        s = self.scale.exp().clamp(max=.1*self.extent)
        scales = torch.cat((s,torch.ones_like(s[:,:1])),1)
        rendered,alpha,normal,_,_,_,_ = rasterization_2dgs(
            means=self.xyz,quats=F.normalize(self.quat,dim=1),scales=scales,
            opacities=self.alpha.sigmoid(),colors=color[None],viewmats=view[None],
            Ks=k[None],width=scene.w,height=scene.h,packed=False,distloss=False,
            render_mode="RGB+ED",backgrounds=self.background[None])
        return rendered[0,:,:,:3], rendered[0,:,:,3], normal[0], alpha[0,:,:,0]


def cameras_for(scene):
    result={}
    for t in scene.names:
        view=torch.eye(4,device="cuda")
        r=torch.from_numpy(scene.R[t]).cuda()
        c=torch.from_numpy(scene.C[t]).cuda()
        view[:3,:3], view[:3,3] = r, -r@c
        result[t]=(view,torch.from_numpy(scene.K[t]).cuda(),c)
    return result


def initialize(views,extent,count):
    parts=[]
    for t in views.ids:
        j=views.slot[t]
        pix=torch.randint(views.w*views.h,(max(1,count//len(views.ids)),),device="cuda")
        parts.append(points_from_pixels(views.scene,t,views.depth[j,0],pix,
                                       views.images[j].float()/255,extent))
    return [torch.cat(items) for items in zip(*parts)]


@torch.no_grad()
def score_field(field,scene,cameras,targets,out):
    rows=[]
    for t in targets:
        rgb,depth,normal,_ = field(scene,t,cameras)
        pred=rgb.clamp(0,1).cpu().numpy()
        gt=scene.image(t).astype("f")/255
        rows.append(dict(view=t,psnr=psnr(pred,gt),ssim=ssim(pred,gt)))
        if t==targets[len(targets)//2]:
            cv2.imwrite(str(out/"development.png"),np.rint(np.concatenate((pred,gt),1)*255).astype("uint8"))
            np.savez_compressed(out/"geometry.npz",depth=depth.cpu().numpy(),normal=normal.cpu().numpy(),view=t)
    return rows


def save(path,obj):
    tmp=path.with_suffix(".tmp")
    torch.save(obj,tmp)
    os.replace(tmp,path)


def train(args,scene,out):
    seed=441
    rng=random.Random(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    dev=sorted(scene.train)[3::19][:12]
    views=Views(scene,scene.train-set(dev))
    views.build_depth()
    extent=float(np.median(np.linalg.norm(scene.pts-np.median(scene.pts,axis=0),axis=1)))
    field=Field(initialize(views,extent,args.points),extent)
    cams=cameras_for(scene)
    rates={"xyz":extent*.00012,"dc":.006,"sh":.0012,"alpha":.018,"quat":.001,"scale":.005,"background":.003}
    opt=torch.optim.Adam([{"params":[p],"lr":rates[n],"name":n} for n,p in field.named_parameters()],
                         eps=1e-12,fused=True)
    start,best,history=1,-float("inf"),[]
    if (out/"resume.pt").is_file():
        ck=torch.load(out/"resume.pt",weights_only=False)
        field.load_state_dict(ck["field"])
        opt.load_state_dict(ck["opt"])
        start,best,history=ck["step"]+1,ck["best"],ck["history"]
        rng.setstate(ck["rng"])
        torch.cuda.set_rng_state(ck["cuda"].cpu())
    for step in range(start,args.steps+1):
        t=rng.choice(views.ids)
        j=views.slot[t]
        gt=views.images[j].permute(1,2,0).float()/255
        rgb,_,_,_=field(scene,t,cams)
        err=rgb-gt
        loss=err.abs().mean()+.4*err.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for g in opt.param_groups:
            g["lr"]=rates[g["name"]]*math.exp(-step/args.steps*(2.5 if g["name"]=="xyz" else 1.1))
        opt.step()
        # Recycle nearly invisible surfels to cover high-error image regions.
        if 600 <= step <= args.steps*.8 and step%80==0:
            with torch.no_grad():
                count=min(650,len(field.alpha)//120)
                index=torch.topk(field.alpha,count,largest=False).indices
                pix=torch.multinomial((err.detach().abs().mean(-1)+.002).square().flatten(),count)
                vals=points_from_pixels(scene,t,views.depth[j,0],pix,gt.permute(2,0,1),extent)
                for param,value in zip((field.xyz,field.dc,field.quat,field.scale),vals):
                    param[index]=value
                field.sh[index]=0
                field.alpha[index]=-.5
                for group in opt.param_groups:
                    p=group["params"][0]
                    if p is not field.background:
                        for key in ("exp_avg","exp_avg_sq"):
                            opt.state[p][key][index]=0
        if step%1000==0 or step==args.steps:
            rows=score_field(field,scene,cams,dev,out)
            score=float(np.mean([r["psnr"] for r in rows]))
            history.append(dict(step=step,score=score,views=rows))
            if score>best:
                best=score
                save(out/"best.pt",dict(field=field.state_dict(),extent=extent,step=step))
            save(out/"resume.pt",dict(field=field.state_dict(),opt=opt.state_dict(),step=step,
                                      best=best,history=history,rng=rng.getstate(),cuda=torch.cuda.get_rng_state()))
            print(scene.root.name[:4],step,round(score,3),flush=True)
            (out/"development.json").write_text(json.dumps(history,indent=2))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("scene",type=Path)
    cli.add_argument("--configs",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--steps",type=int,default=8000)
    cli.add_argument("--points",type=int,default=155000)
    args=cli.parse_args()
    out=args.out/args.scene.name[:4]
    out.mkdir(exist_ok=True,parents=True)
    cv2.setNumThreads(2)
    train(args,Scene(args.scene,args.configs/(args.scene.name+".yaml")),out)
