import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8))

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor='black', text_color='black', fontsize=10):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, horizontalalignment='center', verticalalignment='center', color=text_color, fontsize=fontsize, fontweight='bold', zorder=3)

def draw_ellipse(ax, x, y, width, height, text, facecolor, edgecolor='black', text_color='black', fontsize=10):
    ellipse = patches.Ellipse((x + width/2, y + height/2), width, height, linewidth=1, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(ellipse)
    ax.text(x + width/2, y + height/2, text, horizontalalignment='center', verticalalignment='center', color=text_color, fontsize=fontsize, fontweight='bold', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color='black', linestyle='-', label=None, label_offset_y=0.2, label_offset_x=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='->', lw=1.5, ls=linestyle), zorder=1)
    if label:
        ax.text((x1+x2)/2 + label_offset_x, (y1+y2)/2 + label_offset_y, label, color=color, ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=3)

# Coordinates
w, h = 2.2, 1.2
input_x, input_y = 1, 3.5
trunk_x, trunk_y = 4.5, 3.5

heads_x = [9, 9, 9, 9]
heads_y = [7, 5, 3, 1]

# Draw Input and Trunk
draw_box(ax, input_x, input_y, w, h, "Input Features\n(17-D)", "#EAEAEA")
draw_box(ax, trunk_x, trunk_y, w, h, "Shared Global Trunk\n(64-D)", "#D0E4F5")

draw_arrow(ax, input_x+w, input_y+h/2, trunk_x, trunk_y+h/2)

# Draw Heads
head_labels = ["Diabetes Head\n(32-D)", "CHF Head\n(32-D)", "Sepsis Head\n(32-D)", "AKI Head\n(32-D)"]
logit_labels = ["Diabetes\nLogit", "CHF\nLogit", "Sepsis\nLogit", "AKI\nLogit"]

for i in range(4):
    draw_box(ax, heads_x[i], heads_y[i], w, h, head_labels[i], "#E2F0D9")
    # Draw logits
    draw_ellipse(ax, heads_x[i]+3.5, heads_y[i]+0.1, 1.5, 1.0, logit_labels[i], "white")
    draw_arrow(ax, trunk_x+w, trunk_y+h/2, heads_x[i], heads_y[i]+h/2)
    draw_arrow(ax, heads_x[i]+w, heads_y[i]+h/2, heads_x[i]+3.5, heads_y[i]+h/2)

# Draw Stop Gradients
for i in range(3):
    draw_arrow(ax, heads_x[i]+w/2, heads_y[i], heads_x[i+1]+w/2, heads_y[i+1]+h, color='#D32F2F', linestyle='--', label="Stop-Gradient\n(tensor.detach)", label_offset_x=0.5, label_offset_y=-0.2)

ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
plt.title("DAG Multi-Task Learning Architecture", fontsize=16, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig("../results/Architecture_Diagram.png", dpi=300, bbox_inches='tight', facecolor='white')
print("SUCCESS")
