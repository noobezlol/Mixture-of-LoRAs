import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
colors = {
    'user': '#4CAF50',
    'router': '#2196F3',
    'code': '#FF9800',
    'math': '#9C27B0',
    'base': '#607D8B',
    'lora': '#F44336',
    'base_model': '#795548', # Brown
    'text': 'white',
    'arrow': '#333333',
    'label_bg': '#333333'
}

# Unified Label Style (White text, Semi-transparent dark background)
# This satisfies "keep the background same as others" (semi-transparent)
# while keeping the text opaque/white.
unified_label_style = dict(boxstyle="round,pad=0.3", fc=colors['label_bg'], ec="none", alpha=0.7)

# --- Title ---
ax.text(7, 11.2, 'Mixture-of-LoRAs Architecture', ha='center', va='center',
        fontsize=22, fontweight='bold', color='white', bbox=unified_label_style)

# Subtitle
ax.text(7, 10.5, 'Multi-Expert AI System with Specialized LoRA Adapters', ha='center', va='center',
        fontsize=12, color='white', bbox=unified_label_style)

# --- User Query Box ---
user_box = FancyBboxPatch((5, 9.0), 4, 1,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['user'],
                         edgecolor='none',
                         linewidth=2)
ax.add_patch(user_box)
ax.text(7, 9.5, 'User Query', ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'])

# --- Smart Router Box ---
router_box = FancyBboxPatch((5.5, 7.0), 3, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['router'],
                           edgecolor='none',
                           linewidth=2)
ax.add_patch(router_box)
ax.text(7, 7.6, 'Smart Router\n(Text Classification)', ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'])


# --- Stacks ---

# Left Stack: Code Adapter + Base Model
# Code Adapter Box
code_adapter_box = FancyBboxPatch((1, 4.5), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['code'],
                         edgecolor='none',
                         linewidth=2)
ax.add_patch(code_adapter_box)
ax.text(2.5, 5.1, 'Code Adapter', ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'])

# Plus Sign (White with Unified Style)
ax.text(2.5, 3.9, '+', ha='center', va='center', fontsize=24, fontweight='bold',
        color='white', bbox=unified_label_style)

# Base Model Box (Left)
base_left_box = FancyBboxPatch((1, 2.0), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['base_model'],
                         edgecolor='none',
                         linewidth=2)
ax.add_patch(base_left_box)
ax.text(2.5, 2.6, 'Base Model', ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'])


# Center Stack: Math Adapter + Base Model
# Math Adapter Box
math_adapter_box = FancyBboxPatch((5.5, 4.5), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['math'],
                         edgecolor='none',
                         linewidth=2)
ax.add_patch(math_adapter_box)
ax.text(7, 5.1, 'Math Adapter', ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'])

# Plus Sign (White with Unified Style)
ax.text(7, 3.9, '+', ha='center', va='center', fontsize=24, fontweight='bold',
        color='white', bbox=unified_label_style)

# Base Model Box (Center)
base_center_box = FancyBboxPatch((5.5, 2.0), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['base_model'],
                         edgecolor='none',
                         linewidth=2)
ax.add_patch(base_center_box)
ax.text(7, 2.6, 'Base Model', ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'])


# Right Stack: Base Model Alone
base_right_box = FancyBboxPatch((10, 4.5), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['base_model'],
                         edgecolor='none',
                         linewidth=2)
ax.add_patch(base_right_box)
ax.text(11.5, 5.1, 'Base Model', ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'])


# --- Arrows ---
arrow_props = dict(arrowstyle='->', lw=2, color=colors['arrow'])

# User to Router
ax.annotate('', xy=(7, 8.2), xytext=(7, 9.0),
           arrowprops=arrow_props)

# Router to Stacks (Top Boxes)
# To Code Adapter
ax.annotate('', xy=(2.5, 5.7), xytext=(6, 7.0),
           arrowprops=arrow_props)
# To Math Adapter
ax.annotate('', xy=(7, 5.7), xytext=(7, 7.0),
           arrowprops=arrow_props)
# To Base Model (General)
ax.annotate('', xy=(11.5, 5.7), xytext=(8, 7.0),
           arrowprops=arrow_props)


# --- Labels on Lines ---
# Code Query
ax.text(4.0, 6.5, 'Code Query', ha='center', va='center', fontsize=10,
        color='white', bbox=unified_label_style)

# Math Query
ax.text(7.0, 6.5, 'Math Query', ha='center', va='center', fontsize=10,
        color='white', bbox=unified_label_style)

# General Query
ax.text(10.0, 6.5, 'General Query', ha='center', va='center', fontsize=10,
        color='white', bbox=unified_label_style)

plt.tight_layout()
plt.savefig('architecture_diagram_v4.png', dpi=300, bbox_inches='tight', transparent=True)
# plt.show()
