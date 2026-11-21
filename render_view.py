"""Render from a saved checkpoint using only the training reference images."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment import Scene
from fusion import Renderer,Fusion,variants


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("scene",type=Path)
    cli.add_argument("--configs",type=Path,required=True)
    cli.add_argument("--runs",type=Path,default=Path("runs"))
    camera=cli.add_mutually_exclusive_group(required=True)
    camera.add_argument("--view",type=int)
    camera.add_argument("--camera-json",type=Path,help="K in output pixels, world-to-camera R and world-space center C")
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    scene=Scene(args.scene,args.configs/(args.scene.name+".yaml"))
    t=args.view
    if args.camera_json:
        pose=json.loads(args.camera_json.read_text())
        t=max(scene.names)+1
        scene.names[t]="query"
        for dest,key,shape in ((scene.K,"K",(3,3)),(scene.R,"R",(3,3)),(scene.C,"C",(3,))):
            value=np.array(pose[key],dtype="f")
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError("Invalid camera parameter "+key)
            dest[t]=value
    elif t not in scene.test:
        raise ValueError("Choose an official held-out camera or supply a new camera pose")
    ck=torch.load(args.runs/"fusion"/args.scene.name[:4]/"fusion.pt",weights_only=True)
    renderer=Renderer(scene,scene.train,args.runs)
    net=Fusion().cuda().eval()
    net.load_state_dict(ck["net"])
    with torch.no_grad():
        feat,depth,normal=renderer.render(t)
        rgb=variants(feat,net)[ck["choice"]].permute(1,2,0).cpu().numpy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(args.output),np.rint(rgb*255).clip(0,255).astype("uint8")):
        raise OSError(args.output)
    np.savez_compressed(args.output.with_suffix(".npz"),gaussian_depth=depth.cpu().numpy(),
                        gaussian_normal=normal.cpu().numpy(),K=scene.K[t],R=scene.R[t],C=scene.C[t])


if __name__=="__main__":
    main()
