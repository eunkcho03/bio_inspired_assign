from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ascii_map = """
XXXXXXXXXX
XH.......X
X..1..W..X
X....D...X
X..P.....X
X....2...X
X.....T..X
X..W...3.X
X....4.P.X
XXXXXXXXXX
""".strip("\n")

COLORS = {
    ' ': (0.90, 0.90, 0.90),  
    'X': (0.12, 0.12, 0.12),  
    'T': (0.78, 0.24, 0.24),  
    'W': (0.31, 0.63, 0.86), 
    'D': (0.78, 0.47, 0.24), 
    'P': (0.51, 0.27, 0.78),
    'H': (0.24, 0.71, 0.35), 
    '.': (0.90, 0.90, 0.90),  
}

lines = ascii_map.split("\n")
H = len(lines)
W = len(lines[0])

fig, ax = plt.subplots(figsize=(W * 0.5, H * 0.5))  
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.invert_yaxis() 

for r in range(H):
    for c in range(W):
        ch = lines[r][c]
        if ch.isdigit():
            color = COLORS['.']
        else:
            color = COLORS.get(ch, COLORS['.'])

        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor='black', linewidth=0.8))
        ax.text(c + 0.5, r + 0.55, ch, ha='center', va='center', fontsize=10, color='black')

ax.set_xticks(range(W))
ax.set_yticks(range(H))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("ascii_background.png", dpi=200)
plt.show()
