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
plot_multiple_diagrams(file_names, labels, title, "img/small_bach_38_4.png", scales=scales)