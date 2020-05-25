from utils.generate_figures import plot_multiple_diagrams, moving_average

# TODO move files from source to root dir

##############
# Bach small #
##############
file_names = ["logs/bach_38_4/run-version_21-tag-loss.csv",
                "logs/bach_38_4/run-version_22-tag-loss.csv"]
labels = ["Chunk size 16", "Chunk size 4"]
title = "Training loss for two data points with loss scaling"
scales = [4, 1]
plot_multiple_diagrams(file_names, labels, title, "img/small_bach_38_4.png", scales=scales)

file_names = ["logs/bach_38_4/run-small_bach_chunk16-tag-loss.csv",
                "logs/bach_38_4/run-small_bach_chunk4-tag-loss.csv"]
labels = ["Chunk size 16", "Chunk size 4"]
title = "Training loss for two data points without loss scaling"
scales = [4, 1]
plot_multiple_diagrams(file_names, labels, title, "img/small_bach_38_4_without_scaling.png", scales=scales)

#############
# Scarletti #
#############

file_names = ["logs/layers/run-version_0-tag-loss.csv",
                "logs/layers/run-version_23-tag-loss.csv",
                "logs/layers/run-version_24-tag-loss.csv"]
labels = ["6 layers", "1 layer", "3 layers"]
title = "Training loss for the Scarletti dataset for different models with loss scaling"
scales = [1, 1, 1]
plot_multiple_diagrams(file_names, labels, title, "img/scarletti_layers.png", scales=scales, N=10)