import collections
import itertools

import numpy as np

import igraph
from lingpy.algorithm.clustering import mcl, flat_cluster


def igraph_clustering(matrix, threshold, method='infomap'):
    """
    Wrapper around several of igraph's community structure finding algorithm
    implementations, most notably InfoMap.

    This is originally taken from and builds on LingPy's infomap_clustering
    function.
    """
    G = igraph.Graph()
    vertex_weights = []
    for i in range(len(matrix)):
        G.add_vertex(i)
        vertex_weights += [0]

    # variable stores edge weights, if they are not there, the network is
    # already separated by the threshold
    weights = None
    for i,row in enumerate(matrix):
        for j,cell in enumerate(row):
            if i < j:
                if cell <= threshold:
                    G.add_edge(i, j, weight=1-cell, distance=cell)
                    weights = 'weight'

    if method == 'infomap':
        comps = G.community_infomap(edge_weights=weights,
                vertex_weights=None)

    elif method == 'labelprop':
        comps = G.community_label_propagation(weights=weights,
                initial=None, fixed=None)

    elif method == 'ebet':
        dg = G.community_edge_betweenness(weights=weights)
        oc = dg.optimal_count
        comps = False
        while oc <= len(G.vs):
            try:
                comps = dg.as_clustering(dg.optimal_count)
                break
            except:
                oc += 1
        if not comps:
            print('Failed...')
            comps = list(range(len(G.sv)))
            input()
    elif method == 'multilevel':
        comps = G.community_multilevel(return_levels=False)
    elif method == 'spinglass':
        comps = G.community_spinglass()

    D = {}
    for i, comp in enumerate(comps.subgraphs()):
        vertices = [v['name'] for v in comp.vs]
        for vertex in vertices:
            D[vertex] = i

    return D


def clustering(matrix, threshold, cluster_method='infomap'):
    if callable(cluster_method):
        c = external_function(
            threshold, matrix)
    elif cluster_method in ['infomap', 'labelprop', 'ebet', 'multilevel', 'spinglass']:
        c = igraph_clustering(
            matrix, threshold, method=cluster_method)
    elif cluster_method == 'mcl':
        kw = dict(max_steps=1000,
                  inflation=2,
                  expansion=2,
                  add_self_loops=True,
                  mcl_logs=lambda x: -np.log2((1 - x) ** 2))
        c = mcl(
            threshold, matrix,
            taxa = list(range(len(matrix))), revert=True,
            # More mcl parameters
            max_steps=kw['max_steps'],
            inflation=kw['inflation'],
            expansion=kw['expansion'],
            add_self_loops=kw['add_self_loops'],
            logs=kw['mcl_logs'])
    elif cluster_method in ['upgma', 'single', 'complete', 'ward']:
        c = flat_cluster(
            cluster_method,
            threshold, [[c for c in r] for r in matrix],
            revert=True)
    else:
        raise ValueError("No clustering method named {:}".format(cluster_method))
    return c


def cluster(dataset, scores, threshold=0.5, method='infomap'):
    """
    Cluster the dataset's synonymous words into cognate sets based on distance
    scores between each pair of words. Return a dict mapping concepts to frozen
    sets of frozen sets of Word tuples.

    The second arg should be a {(word1, word2): distance} dict where the keys
    are sorted Word named tuples tuples. The keyword args are passed on to the
    InfoMap algorithm.
    """
    clusters = {}

    for concept, words in dataset.get_concepts().items():
        if len(words) <= 1: continue

        matrix = np.zeros((len(words), len(words),))

        for (i, word1), (j, word2) in itertools.combinations(enumerate(words), 2):
            key = (word1, word2) if word1 < word2 else (word2, word1)
            try:
                matrix[j, i] = matrix[i, j] = scores[key]
            except KeyError:
                key = (word1, word2) if word1 > word2 else (word2, word1)
                matrix[j, i] = matrix[i, j] = scores[key]

        index_labels = clustering(matrix, threshold, method)

        cog_sets = collections.defaultdict(set)
        for index, label in index_labels.items():
            cog_sets[label].add(words[index])

        clusters[concept] = frozenset([frozenset(s) for s in cog_sets.values()])

    return clusters
