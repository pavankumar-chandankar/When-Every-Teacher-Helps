"""Dataset-independent, vector teaser for PerFail 2027.

Conceptual workflow only. No numeric or as-yet-unrun cross-dataset results.
Run: python3 scripts/make_teaser_overview.py
Output: fig_teaser.pdf (editable generator included in the Overleaf bundle).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle

OUT = Path(__file__).resolve().parent.parent / 'fig_teaser.pdf'
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.transparent': False,
})
NAVY='#172C3E'
MUTED='#587080'
BLUE='#1965A5'
BLUELIGHT='#EDF5FC'
CORAL='#B64E42'
CORALLIGHT='#FDF2EF'
TEAL='#147B75'
TEALLIGHT='#ECF7F4'
GRAY='#F5F7F9'
BORDER='#CDD9E1'
FIGW, FIGH=3.48, 2.16
fig=plt.figure(figsize=(FIGW,FIGH), facecolor='white')
ax=fig.add_axes((0,0,1,1))
ax.set_xlim(0,FIGW); ax.set_ylim(0,FIGH); ax.axis('off')

def box(x,y,w,h,face,edge=BORDER,r=0.075,lw=.75):
    patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.00,rounding_size={r}',
        fc=face,ec=edge,lw=lw,mutation_aspect=1,zorder=1)
    ax.add_patch(patch)

def label(x,y,text,sz,color=NAVY,weight='normal',ha='center'):
    ax.text(x,y,text,fontsize=sz,color=color,fontweight=weight,
            va='center',ha=ha,zorder=3)

def arrow(x1,y1,x2,y2,color=MUTED,lw=1.35,ms=8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
        color=color,linewidth=lw,mutation_scale=ms,shrinkA=0,shrinkB=0,zorder=4))

# Title strip is intentionally a question, not an outcome.
label(.05,2.015,'DOES AN ACCURACY GAIN MEAN KNOWLEDGE TRANSFER?',7.35,NAVY,'bold','left')
ax.plot([.04,3.44],[1.895,1.895],color=BORDER,lw=.65)

# Three-stage flow: observation -> intervention -> apparent result.
fy,fh=1.12,.67
box(.04,fy,1.02,fh,BLUELIGHT,'#8EBDD9')
box(1.235,fy,.99,fh,GRAY,BORDER)
box(2.405,fy,1.03,fh,TEALLIGHT,'#9ACDC4')

# Tiny WiFi arcs and synchronized modalities icon, all drawn as vector primitives.
for radius in (.055,.085,.12):
    ax.add_patch(Arc((.17,1.485),radius*2,radius*1.6,
        theta1=35,theta2=145,ec=BLUE,lw=.95,zorder=3))
ax.add_patch(Circle((.17,1.424),.011,fc=BLUE,ec='none',zorder=3))
label(.67,1.605,'PAIRED SIGNALS',6.35,BLUE,'bold')
label(.67,1.447,'WiFi + IMU',7.7,NAVY,'bold')
label(.67,1.286,'pose + video',6.7,MUTED)

label(1.73,1.605,'TRAIN',6.3,MUTED,'bold')
label(1.73,1.447,'Cross-modal',7.65,NAVY,'bold')
label(1.73,1.286,'supervision',7.65,NAVY,'bold')

label(2.92,1.605,'OBSERVE',6.3,TEAL,'bold')
label(2.92,1.445,'Higher score',7.85,NAVY,'bold')
label(2.92,1.288,'on validation',6.6,MUTED)

arrow(1.082,1.456,1.214,1.456)
arrow(2.247,1.456,2.386,1.456)

label(1.74,.990,'CHECK THE EXPLANATION, NOT JUST THE SCORE',6.35,NAVY,'bold')
ax.plot([.585,2.92],[.915,.915],lw=.85,color=TEAL,zorder=3)
for x in (.585,1.745,2.915):
    ax.plot([x,x],[.915,.877],lw=.85,color=TEAL,zorder=3)

cy,ch=.275,.602
for x in (.04,1.205,2.37):
    box(x,cy,1.07,ch,'white',BORDER)
# Narrow colored indicator to identify independent audits.
for x,color in ((.06,CORAL),(1.225,BLUE),(2.39,TEAL)):
    ax.plot([x,x],[.395,.747],color=color,lw=2.6,solid_capstyle='round')

for x,head,second in (
    (.59,'Teacher','specificity?'),
    (1.75,'Motion','observable?'),
    (2.92,'Unseen-room','robustness?'),
):
    label(x,.657,head,7.15,NAVY,'bold')
    label(x,.473,second,7.1,NAVY)

# Result-neutral caution reinforces that this is a *study design*.
box(.04,.05,3.39,.153,'#EEF2F5','none',.035)
label(1.735,.126,'VERIFY MEASUREMENTS WITH KNOWN-ANSWER TESTS',6.45,MUTED,'bold')

fig.savefig(OUT,dpi=300,pad_inches=0)
print('WROTE',OUT)
