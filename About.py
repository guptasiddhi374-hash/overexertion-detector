import matplotlib.pyplot as plt

# --- For your Confusion Matrix ---
# (Assume 'cm_display' is your confusion matrix plot)
cm_display.plot() 
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# --- For your Learning Curve ---
# (Assume you just plotted your learning curve)
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()