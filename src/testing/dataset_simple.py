import matplotlib.pyplot as plt

from src.graph_utils.Graph import Graph, Mode
from src.evolution.mutations import \
    mab  # or lin2opt, double_bridge, point_exchange, couple_exchange, relocate_block, combined
from src.evolution.run_evolution import run_evolution
from src.visualization.draw_graph import draw_graph

if __name__ == "__main__":
    g = Graph(Mode.IDS, dataset_dir="testing")
    g.set_path_weight(g.get_path_weight())

    evolution = run_evolution(n_epochs=20, pool_size=20, mutation=mab, initial_graph=g, gif_filename='test')

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(5, 15))

    ax0.plot(evolution.get_fitness_hist())
    ax0.set_title('fitness hist')
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('fitness')

    ax1.set_facecolor('#fcfcfc')
    draw_graph(ax1, g)
    ax1.set_title('start path')
    ax1.set_xlabel('x coord')
    ax1.set_ylabel('y coord')

    ax2.set_facecolor('#fcfcfc')
    best_path = evolution.get_best_path()
    draw_graph(ax2, best_path)
    ax2.set_title('result path')
    ax2.set_xlabel('x coord')
    ax2.set_ylabel('y coord')

    plt.show()
