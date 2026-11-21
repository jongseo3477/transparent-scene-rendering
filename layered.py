"""Calibrated multi-view stereo and learned two-depth RGB compositing on CUDA."""
import argparse
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment import Scene, psnr, ssim


class Views:
    def __init__(self, scene, references):
        self.scene = scene
        self.ref = references
        self.ids = sorted(references)
        self.slot = {idx: j for j, idx in enumerate(self.ids)}
        self.images = torch.as_tensor(np.stack([scene.image(i) for i in self.ids]),
                                      device="cuda").permute(0, 3, 1, 2).contiguous()
        self.poses = {}
        self.depth = None
        self.h, self.w = scene.h, scene.w

    def pose(self, t):
        if t not in self.poses:
            scene = self.scene
            _, far = scene.sources(t, self.ref-{t})
            scale = np.median(np.linalg.norm(scene.pts - scene.C[t], axis=1))
            def dist(i):
                return np.linalg.norm(scene.C[i]-scene.C[t])/scale + .55*np.linalg.norm(scene.R[i]-scene.R[t])
            src = sorted(self.ref-{t}, key=dist)[:8]
            mat, offset = [], []
            for i in src:
                mat.append(scene.K[i] @ scene.R[i] @ scene.R[t].T @ np.linalg.inv(scene.K[t]))
                offset.append(scene.K[i] @ scene.R[i] @ (scene.C[t]-scene.C[i]))
            self.poses[t] = (torch.tensor([self.slot[i] for i in src],device="cuda"),
                             torch.as_tensor(np.stack(mat),device="cuda"),
                             torch.as_tensor(np.stack(offset),device="cuda"), float(far))
        return self.poses[t]

    def rays(self,t, box, stride=1):
        x,y,w,h = box
        ys,xs = torch.meshgrid(torch.arange(y,y+h,stride,device="cuda"),
                               torch.arange(x,x+w,stride,device="cuda"),indexing="ij")
        pix = torch.stack((xs,ys,torch.ones_like(xs))).float()
        ids, mat, off, far = self.pose(t)
        uv = (mat @ pix.flatten(1)).reshape(len(ids),3,*xs.shape)
        return ids,uv,off,far

    def project(self, rays, d, src):
        _,uv,b,_ = rays
        p = uv[:,None] + b[:,None,:,None,None]*d[None,:,None]
        z = p[:,:,2].clamp_min(.001)
        u = p[:,:,0]/z * (2/(self.w-1))-1
        v = p[:,:,1]/z * (2/(self.h-1))-1
        valid = (u.abs()<=1) & (v.abs()<=1) & (p[:,:,2]>.001)
        g = torch.stack((u,v),-1)
        n,k,h,w,_ = g.shape
        val = F.grid_sample(src, g.reshape(n,k*h,w,2),padding_mode="border",
                            align_corners=True).reshape(n,-1,k,h,w)
        return val,valid,z

    @torch.no_grad()
    def build_depth(self):
        priors = []
        for t in self.ids:
            rays = self.rays(t,(0,0,self.w,self.h),stride=2)
            ids,_,_,far = rays
            rgb = self.images[ids].float()/255
            gt = self.images[self.slot[t],:,::2,::2].float()/255
            costs = []
            for d in torch.linspace(0,far,72,device="cuda").split(6):
                colors,mask,_ = self.project(rays,d[:,None,None],rgb)
                dif = (colors-gt[None,:,None]).abs().mean(1).clamp(max=.25)
                dif = torch.where(mask,dif,.3)
                dif = F.avg_pool2d(dif,3,stride=1,padding=1,count_include_pad=False)
                costs.append(dif.topk(4,dim=0,largest=False).values.mean(0))
            costs = torch.cat(costs)
            best = costs.argmin(0,keepdim=True)
            center = costs.gather(0,best)
            a = costs.gather(0,(best-1).clamp_min(0))
            b = costs.gather(0,(best+1).clamp_max(71))
            frac = ((a-b)/(2*(a+b-2*center).clamp_min(1e-5))).clamp(-.5,.5)
            depth = ((best+frac)*far/71).clamp_min(0)
            prior = F.interpolate(depth[None],(self.h,self.w),mode="bilinear",align_corners=False)[0]
            priors.append(prior)
        self.depth = torch.stack(priors)


class Compositor(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Conv3d(8,12,3,padding=1),nn.SiLU(),
                                   nn.Conv3d(12,8,3,padding=1),nn.SiLU(),nn.Conv3d(8,2,1))
        self.blend = nn.Sequential(nn.Conv2d(13,32,3,padding=1),nn.SiLU(),
                                   nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,4,3,padding=1))
        for layer in (self.score[-1], self.blend[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, data,t,box):
        rays = data.rays(t,box)
        ids,_,_,far = rays
        src = torch.cat((data.images[ids].float()/255,data.depth[ids]),1)
        ds = torch.linspace(0,far,56,device="cuda")[:,None,None]
        val, valid,z = data.project(rays,ds,src)
        colors = val[:,:3]
        err = (val[:,3] - ds/z).abs() / far
        def mix(col, valid, err):
            iw = torch.tensor([1,.94,.76,.65,.52,.43,.35,.3],device=col.device)[:,None,None,None]
            w = iw * valid * (.012 + torch.exp(-err*38))
            return (col*w[:,None]).sum(0)/w.sum(0).clamp_min(1e-6)
        mean = mix(colors,valid,err)
        dif = (colors-mean).abs().mean(1)
        near = torch.where(valid,err,.9).topk(3,dim=0,largest=False).values
        photo = torch.where(valid,dif,.4).topk(4,dim=0,largest=False).values
        feats = torch.stack((near[0],near[1],near[2],photo[0],photo[1],photo[2],
                             photo[3],valid.float().mean(0)),0).clamp(0,1)
        coarse = -70*(near[0]*.2+near[1]*.5+near[2]*.3)-45*photo[1:4].mean(0)
        delta = self.score(feats[None]*4)[0].float()
        # Two depth hypotheses: agreement of several views, and photometric match.
        logits = torch.stack((coarse,-75*photo[1:4].mean(0)-20*near[1])) + delta
        prob = logits.softmax(1)
        idx = logits.argmax(1)
        bins = torch.arange(len(ds),device="cuda")[:,None,None]
        local = prob * ((bins-idx[:,None]).abs()<=3)
        depth = (local*ds).sum(1)/local.sum(1).clamp_min(1e-9)
        out,m,z = data.project(rays,depth,src)
        err2 = (out[:,3] - depth/z).abs()/far
        rgb = mix(out[:,:3],m,err2).permute(1,0,2,3)
        broad = (mean * prob[0]).sum(1)
        inputs = torch.cat((rgb[0],rgb[1],broad,(rgb[0]-rgb[1]).abs(),prob[0].amax(0,keepdim=True)))
        residual = self.blend(inputs[None])[0].float()
        alpha = torch.sigmoid(residual[3:4]-1.2)
        return (1-alpha)*rgb[0] + alpha*rgb[1] + residual[:3]*.15, depth, alpha


@torch.no_grad()
def evaluation(data, model, targets, out, tag):
    rows = []
    for j,t in enumerate(targets):
        pred = np.empty((data.h,data.w,3),"f")
        for y in range(0,data.h,64):
            for x in range(0,data.w,96):
                a,b=max(0,x-5),max(0,y-5)
                ex,ey=min(data.w,x+96),min(data.h,y+64)
                box=(a,b,min(data.w,ex+5)-a,min(data.h,ey+5)-b)
                rgb,_,_ = model(data,t,box)
                pred[y:ey,x:ex] = rgb[:,y-b:ey-b,x-a:ex-a].permute(1,2,0).cpu().numpy()
        pred = pred.clip(0,1)
        gt = data.scene.image(t).astype("f")/255
        rows.append(dict(view=t,psnr=psnr(pred,gt),ssim=ssim(pred,gt)))
        if j==len(targets)//2:
            cv2.imwrite(str(out/(tag+".png")),np.rint(np.concatenate((pred,gt),1)*255).astype("uint8"))
    return rows


def run(args):
    torch.manual_seed(2026)
    torch.backends.cudnn.allow_tf32 = True
    rng = random.Random(2026)
    scene = Scene(args.scene,args.configs/(args.scene.name+".yaml"))
    out=args.out/args.scene.name[:4]
    out.mkdir(exist_ok=True,parents=True)
    dev=sorted(scene.train)[3::19][:12]
    training=scene.train-set(dev)
    views=Views(scene,training)
    views.build_depth()
    print(scene.root.name[:4],"priors ready",flush=True)
    model=Compositor().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0015)
    history=[]
    best=-float("inf")
    for step in range(1,args.steps+1):
        t=rng.choice(views.ids)
        w=h=40
        x,y=rng.randrange(views.w-w+1),rng.randrange(views.h-h+1)
        rgb,_,_=model(views,t,(x,y,w,h))
        gt=views.images[views.slot[t],:,y:y+h,x:x+w].float()/255
        diff=(rgb-gt)[:,4:-4,4:-4]
        loss=diff.square().mean()+.04*diff.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        if step in {args.steps//2,args.steps}:
            rows=evaluation(views,model,dev,out,"development")
            score=float(np.mean([r["psnr"] for r in rows]))
            history.append(dict(step=step,psnr=score,views=rows))
            if score>best:
                best=score
                torch.save(model.state_dict(),out/"model.pt")
            print(scene.root.name[:4],step,"dev",round(score,3),flush=True)
    report=dict(scene=scene.root.name,development=history,n_train=len(training),official_train=len(scene.train))
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    if args.final:
        model.load_state_dict(torch.load(out/"model.pt",weights_only=True))
        final=Views(scene,scene.train)
        final.build_depth()
        report["test"]=evaluation(final,model,sorted(scene.test),out,"test")
        (out/"metrics.json").write_text(json.dumps(report,indent=2))
        print("test",scene.root.name[:4],np.mean([r["psnr"] for r in report["test"]]),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("scene",type=Path)
    cli.add_argument("--configs",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--steps",type=int,default=1000)
    cli.add_argument("--final",action="store_true")
    cv2.setNumThreads(2)
    run(cli.parse_args())
