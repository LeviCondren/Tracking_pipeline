##### Draw training performance plot (loss purity eff for metric and GNN) #####

# %%
import os

import pandas as pd
from bokeh.io import output_notebook, show, save
from bokeh.plotting import figure, row
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
from bokeh.models.annotations import Label
output_notebook()

def get_training_metrics(log_dir_path):
    log_file = os.path.join(log_dir_path, 'metrics.csv')

    metrics = pd.read_csv(log_file, sep=',')

    train_metrics = metrics[~metrics['train_loss'].isna()][['epoch', 'train_loss']]

    train_metrics['epoch'] -= 1

    val_metrics = metrics[~metrics['val_loss'].isna()][['val_loss', 'eff', 'pur', 'current_lr', 'epoch']]

    metrics = pd.merge(left=train_metrics, right=val_metrics, how='inner', on='epoch')

    return metrics

def plot_training_metrics(metrics):

    p1 = figure(title='Training validation loss', x_axis_label='Epoch', y_axis_label='Loss', y_axis_type="log")

    source = ColumnDataSource(metrics)

    cmap = viridis(3)

    for idx, y in enumerate(['train_loss', 'val_loss']):
        p1.circle(y=y, x='epoch', source=source, color=cmap[idx], legend_label=y)
        p1.line(x='epoch', y=y, source=source, color=cmap[idx], legend_label=y)


    p2 = figure(title='Purity on validation set', x_axis_label='Epoch', y_axis_label='Purity')
    p2.circle(y='pur', x='epoch', source=source, color=cmap[0], legend_label='Purity')
    p2.line(x='epoch', y='pur', source=source, color=cmap[0], legend_label='Purity')

    p3 = figure(title='Efficiency on validation set', x_axis_label='Epoch', y_axis_label='Efficiency')
    p3.circle(y='eff', x='epoch', source=source, color=cmap[0], legend_label='Efficiency')
    p3.line(x='epoch', y='eff', source=source, color=cmap[0], legend_label='Efficiency')

    show(row([p1, p2, p3]))



#log_dir_path_gnn_quirk_100_100 = "artifacts/Lambda500_pre_selection_quirk/gnn/quirk/version_0"
log_dir_path_metric_quirk_100_100 = "artifacts/Lambda500_pre_selection_quirk/metric_learning/quirk/version_0"


#gnn_quirk_100_100 = get_training_metrics(log_dir_path_gnn_quirk_100_100)
metric_quirk_100_100 = get_training_metrics(log_dir_path_metric_quirk_100_100)


#print(gnn_metrics)
print("gnn_pre_quirk_100_100:")
#plot_training_metrics(gnn_quirk_100_100)
print("metric_pre_quirk_100_100:")
plot_training_metrics(metric_quirk_100_100)

