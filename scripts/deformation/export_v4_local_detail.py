"""Crop the same native comparison to the fixed local support; no motion amplification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
from scripts.events.render_event_only_sequence import _ffmpeg_executable


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    mask=np.load(args.run/'local-event-mask.npy').any(0)
    y,x=np.where(mask)
    if len(x)==0:raise ValueError('Empty local region')
    left=max(0,int(x.min()*8)-16);top=max(0,int(y.min()*8)-16)
    right=min(640,int((x.max()+1)*8)+16);bottom=min(640,int((y.max()+1)*8)+16)
    size=max(right-left,bottom-top)
    size=min(640,(size+1)//2*2)
    left=max(0,min(640-size,(left+right-size)//2));top=max(0,min(640-size,(top+bottom-size)//2))
    output=args.run/'local-detail.mp4'
    ffmpeg=_ffmpeg_executable()
    if not ffmpeg:raise RuntimeError('ffmpeg required')
    # Crop excludes the title bar, then re-add clear labels. Each image is an
    # equal crop/zoom of its own original panel; apparent motion is not scaled.
    subprocess.run([ffmpeg,'-nostdin','-v','error','-n','-i',str(args.run/'comparison.mp4'),
                    '-filter_complex',f'[0:v]split=2[a][b];[a]crop={size}:{size}:{left}:{top+32},scale=640:640:flags=lanczos[l];[b]crop={size}:{size}:{left+640}:{top+32},scale=640:640:flags=lanczos[r];[l][r]hstack=inputs=2[out]',
                    '-map','[out]','-an','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    for path in args.run.glob('comparison-*.png'):
        source=Image.open(path).convert('RGB');canvas=Image.new('RGB',(1280,672),(24,24,24));draw=ImageDraw.Draw(canvas)
        for offset,label in ((0,'V3: local detail'),(640,'V4: local MLP detail')):
            crop=source.crop((left+offset,top+32,left+offset+size,top+32+size)).resize((640,640),Image.Resampling.LANCZOS)
            canvas.paste(crop,(offset,32));draw.text((offset+12,10),label,fill='white')
        canvas.save(args.run/path.name.replace('comparison-','local-detail-'))
    (args.run/'local-detail.json').write_text(json.dumps(dict(source='comparison.mp4',output=output.name,
        crop=dict(x=left,y=top,size=size),scope='Left V3, right V4; identical crop and zoom, source frame rate, no amplified motion or generated details'),indent=2)+'\n')


if __name__=='__main__':main()
