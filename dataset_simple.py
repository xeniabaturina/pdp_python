import matplotlib.pyplot as plt

from src.graph_utils.Graph import Graph, Mode
from src.evolution.mutations import \
    mab  # or lin2opt, double_bridge, point_exchange, couple_exchange, relocate_block, combined
from src.evolution.run_evolution import run_evolution
from src.visualization.draw_graph import draw_graph

if __name__ == "__main__":
    g = Graph(Mode.IDS, dataset_dir="data/testing")
    g.set_path_weight(g.get_path_weight())

    evolution = run_evolution(n_epochs=20, pool_size=20, seed=42, mutation=mab, initial_graph=g, gif_filename='test')

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(5, 15))

    fontsize = 10
    for i, ax in enumerate((ax0, ax1, ax2)):
        if i == 0:
            ax.plot(evolution.get_fitness_hist())
            ax.set_title('fitness hist', fontsize=fontsize)
            ax.set_xlabel('epoch', fontsize=fontsize)
            ax.set_ylabel('fitness', fontsize=fontsize)

        else:
            ax.set_facecolor('#fcfcfc')
            if i == 1:
                draw_graph(ax, g)
                ax.set_title('start path', fontsize=fontsize)
            else:
                best_path = evolution.get_best_path()
                draw_graph(ax2, best_path)
                ax.set_title('result path', fontsize=fontsize)

            ax.set_xlabel('x coord', fontsize=fontsize)
            ax.set_ylabel('y coord', fontsize=fontsize)

    plt.show()
