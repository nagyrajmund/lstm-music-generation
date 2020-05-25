from utils.generate_figures import plot_multiple_diagrams, moving_average

# TODO move files from source to root dir

##############
# Bach small #
##############
file_names = ["logs/bach_38_4/run-version_21-tag-loss.csv",
                "logs/bach_38_4/run-version_22-tag-loss.csv"]
labels = ["Chunk size 16", "Chunk size 4"]
title = "Comparison of performance on a small dataset"
scales = [4, 1]
plot_multiple_diagrams(file_names, labels, title, "img/small_bach_38_4.png", scales=scales)