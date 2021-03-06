from utils.generate_figures import plot_multiple_diagrams, moving_average, plot_distribution

################
# Distribution #
################
dictionary = {'wait1': 769227, 'wait2': 144122, 'p26': 50207, 'p27': 50083, 'p25': 50068, 'p24': 49460,
    'p28': 49458, 'p23': 47730, 'p29': 47194, 'p22': 46603, 'p30': 46256, 'endp26': 44277, 'endp27': 44221,
        'endp25': 43991, 'p21': 43772}
plot_distribution(dictionary)

##############
# Bach small #
##############
file_names = ["logs/bach_38_4/run-version_21-tag-loss.csv",
                "logs/bach_38_4/run-version_22-tag-loss.csv"]
labels = ["Chunk size 16", "Chunk size 4"]
title = "Training loss for two data points with loss scaling"
scales = [1/2, 1/8]
plot_multiple_diagrams(file_names, labels, title, "img/small_bach_38_4.png", scales=scales, ylimits=(0, 4.5))

file_names = ["logs/bach_38_4/run-small_bach_chunk16-tag-loss.csv",
                "logs/bach_38_4/run-small_bach_chunk4-tag-loss.csv"]
labels = ["Chunk size 16", "Chunk size 4"]
title = "Training loss for two data points without loss scaling"
scales = [1/2, 1/8]
plot_multiple_diagrams(file_names, labels, title, "img/small_bach_38_4_without_scaling.png", scales=scales, ylimits=(0, 4.5))

#############
# Scarlatti #
#############

file_names = ["logs/layers/run-version_0-tag-loss.csv",
                "logs/layers/run-version_23-tag-loss.csv",
                "logs/layers/run-version_24-tag-loss.csv"]
labels = ["6 layers", "1 layer", "3 layers"]
title = "Training loss for the Scarlatti dataset for different models with loss scaling"
scales = [1/342, 1/386, 1/386]
plot_multiple_diagrams(file_names, labels, title, "img/scarlatti_layers.png", scales=scales, N=10, ylimits=(2, 4.5))

file_names = ["logs/hidden/run-version_27-tag-loss.csv",
                "logs/hidden/run-version_29-tag-loss.csv"]
labels = ["Embedding: 600, hidden: 900", "Embedding: 800, hidden: 1200"]
title = "Training loss for the Scarlatti dataset with different sizes and loss scaling"
scales = [1/342, 1/455]
plot_multiple_diagrams(file_names, labels, title, "img/scarlatti_hidden.png", scales=scales, N=10, ylimits=(2, 4.5))