"""Task diagram in the manuscript's Kaggle-inspired serif style."""
import matplotlib.pyplot as plt
import plot_review_revision as prior

plt.rcParams.update({'font.family':'DejaVu Serif', 'mathtext.fontset':'dejavuserif'})
_original_save = prior.save

def save(fig, name):
    _original_save(fig, 'kaggle_task')

prior.save = save
if __name__ == '__main__':
    prior.task_figure()
