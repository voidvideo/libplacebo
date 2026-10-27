from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
P=Path(sys.argv[1]); P.mkdir(parents=True, exist_ok=True)
W,H=960,540
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
def scene(n,mode):
 im=Image.new('RGB',(W,H),(22,26,35));d=ImageDraw.Draw(im)
 for x in range(0,W,24): d.line((x,0,x,H),fill=(32,38,48))
 for y in range(0,H,24): d.line((0,y,W,y),fill=(32,38,48))
 d.text((30,15),'VIDEO LIBRARY     Movies     Shows     Settings',font=font,fill='white')
 d.rectangle((25,62,935,505),fill=(42,48,62),outline=(110,120,140),width=2)
 if mode=='scroll':
  for k in range(8):
   x=50+k*160-n*48
   d.rectangle((x,110,x+138,355),fill=((70+k*31)%200,(80+k*43)%190,(100+k*23)%210),outline='white',width=2)
   for yy in range(140,330,32): d.line((x+12,yy,x+125,yy+16),fill=(205,205,205),width=3)
   d.text((x+8,370),f'MOVIE {k+1}',font=font,fill='white')
  d.rectangle((210,100,358,410),outline=(60,200,255),width=5)
 else:
  selected=int(n)%5
  for k,title in enumerate(['Picture settings','Audio output','Network connection','Display calibration','System information']):
   y=95+k*72
   d.rectangle((55,y,585,y+57),fill=(40,170,240) if k==selected else (55,61,74))
   d.text((78,y+14),title,font=font,fill='white')
  x=650+int(n*21)%120
  d.rectangle((x,112,x+80,386),fill=(185,50,75),outline='white',width=3)
  d.text((x+7,185),'INFO',font=small,fill='white')
 d.rectangle((0,510,W,H),fill=(10,12,18));d.text((25,516),'BACK     SELECT                         24 fps source / 60 fps output',font=small,fill='white')
 return im.convert('RGBA')
for mode in ['scroll','selection']:
 with (P/f'{mode}.rgba').open('wb') as f:
  for n in range(12): f.write(scene(n,mode).tobytes())
 for n in [0,1]:scene(n,mode).save(P/f'{mode}-source{n}.png')

for mode in ["scroll", "selection"]:
 with (P / f"{mode}-reference.rgba").open("wb") as f:
  for tick in range(28):
   f.write(scene(tick * .4, mode).tobytes())
