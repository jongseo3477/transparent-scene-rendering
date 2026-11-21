"""Blend source-view correspondences transported to a calibrated query camera."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from experiment import grid, remap


class Correspondences:
    def __init__(self):
        self.net=raft_large(weights=Raft_Large_Weights.DEFAULT,progress=False).eval().cuda()

    @torch.no_grad()
    def predict(self,a,b):
        rgb=np.stack([a[...,::-1],b[...,::-1]]).copy()
        pair=torch.from_numpy(rgb).cuda().permute(0,3,1,2).float()/127.5-1
        h,w=a.shape[:2]
        pair=F.pad(pair,(0,(-w)%8,0,(-h)%8),mode="replicate")
        flow=self.net(pair,pair.flip([0]),num_flow_updates=16)[-1][:,:,:h,:w]
        return flow.permute(0,2,3,1).cpu().numpy()

    def render(self,scene,t,allowed):
        src,max_inv=scene.sources(t,allowed)
        pix=grid(scene.h,scene.w)[...,:2].copy()
        k=scene.K[t]
        center=scene.C[t]
        rot=scene.R[t]
        f=np.diag(k)[:2].copy()
        principal=k[:2,2]
        # Prefer two nearby cameras on opposite sides of the query.
        delta={s:rot@(scene.C[s]-center) for s in src}
        def pair_score(pair):
            i,j=pair
            di,dj=delta[i],delta[j]
            v=di-dj
            u=np.clip(di@v/(v@v+1e-9),0,1)
            offset=np.linalg.norm(di-u*v)
            return offset*3+np.linalg.norm(di)+np.linalg.norm(dj)
        pairs=sorted([(i,j) for n,i in enumerate(src) for j in src[n+1:]],key=pair_score)[:3]
        warped={}
        for i in src:
            H=k@rot@scene.R[i].T@np.linalg.inv(scene.K[i])
            im=scene.image(i)
            warped[i]=cv2.warpPerspective(im,H,(scene.w,scene.h),borderMode=cv2.BORDER_REPLICATE)
        colors,weights=[],[]
        for rank,(i,j) in enumerate(pairs):
            flows=self.predict(warped[i],warped[j])
            for side,(a,b) in enumerate(((i,j),(j,i))):
                flow,reverse=flows[side],flows[1-side]
                ab=delta[a]-delta[b]
                at=delta[a]
                qt=pix+flow
                v=f*ab[:2]+(principal-qt)*ab[2]
                denom=(v*v).sum(-1)
                d=((flow*v).sum(-1)/(denom+1e-8)).clip(0,max_inv*2)[...,None]
                def project(m):
                    return (f*m[:2]+(principal-pix)*m[2])*d/(1+d*m[2]).clip(.05)
                fraction=np.clip((at@ab)/(ab@ab+1e-9),-1,2)
                displacement=project(at)+fraction*(flow-project(ab))
                uv=pix.copy()
                for _ in range(5):
                    uv=pix-remap(displacement,uv[...,0],uv[...,1])
                # Remap once from the original image to preserve high frequencies.
                H=scene.K[a]@scene.R[a]@rot.T@np.linalg.inv(k)
                source=cv2.perspectiveTransform(uv,H)
                color=remap(scene.image(a),source[...,0],source[...,1]).astype("f")/255
                loop=np.linalg.norm(flow+remap(reverse,qt[...,0],qt[...,1]),axis=-1)
                match=remap(loop,uv[...,0],uv[...,1])
                reproj=np.linalg.norm(uv+remap(displacement,uv[...,0],uv[...,1])-pix,axis=-1)
                valid=(source[...,0]>=0)&(source[...,0]<=scene.w-1)&(source[...,1]>=0)&(source[...,1]<=scene.h-1)
                weight=np.exp(-.5*np.minimum(match,20)-np.minimum(reproj,10))*valid
                weights.append(weight/((1+rank*.65)*(.3+abs(fraction))))
                colors.append(color)
        weights=np.stack(weights)
        colors=np.stack(colors)
        total=weights.sum(0)
        pred=(weights[...,None]*colors).sum(0)/np.maximum(total[...,None],1e-8)
        pred=np.where(total[...,None]>1.e-5,pred,colors[0])
        variance=(weights[...,None]*abs(colors-pred)).sum(0)/(total[...,None]+1e-6)
        return pred.astype("f"),variance.mean(-1).astype("f")
