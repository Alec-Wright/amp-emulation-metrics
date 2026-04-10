def greedy_vertex_selection(graph, vertex_weights, edge_weights, k):
    """
    Selects k vertices from a graph using a greedy algorithm based on vertex and edge weights.

    Args:
        graph (dict): The graph represented as a dictionary where the keys are vertices and the values are lists of adjacent vertices.
        vertex_weights (dict): The weights of each vertex represented as a dictionary where the keys are vertices and the values are weights.
        edge_weights (dict): The weights of each edge represented as a dictionary where the keys are tuples of vertices and the values are weights.
        k (int): The number of vertices to select.

    Returns:
        set: A set of selected vertices.

    """
    selected_vertices = set()
    remaining_vertices = set(graph.keys())

    while len(selected_vertices) < k:
        min_cost = float('inf')
        best_vertex = None
        
        for v in remaining_vertices:
            cost = vertex_weights[v]
            for u in selected_vertices:
                # check if the edge exists in the graph
                if (v, u) in edge_weights:
                    cost += edge_weights[(v, u)]
                elif (u, v) in edge_weights:
                    cost += edge_weights[(u, v)]
            
            if cost < min_cost:
                min_cost = cost
                best_vertex = v

        if best_vertex is None:
            print('No more suitable verteces found.')
            break
        selected_vertices.add(best_vertex)
        remaining_vertices.remove(best_vertex)
    
    return selected_vertices

if __name__ == '__main__':
    # Example usage
    graph = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2, 4],
        4: [3]
    }
    vertex_weights = {1: 10, 2: 20, 3: 30, 4: 40}
    edge_weights = {(1, 2): 5, (1, 3): 10, (2, 3): 15, (3, 4): 20}
    k = 2

    selected_vertices = greedy_vertex_selection(graph, vertex_weights, edge_weights, k)
    print(selected_vertices)

def drop_vertex(graph, vertex):
    """
    Drops a vertex from a graph by removing it from the graph and all adjacency lists.

    Args:
        graph (dict): The graph represented as a dictionary where the keys are vertices and the values are lists of adjacent vertices.
        vertex: The vertex to drop.

    Returns:
        dict: The updated graph with the vertex removed.

    """
    if vertex in graph:
        del graph[vertex]
    
    for v in graph:
        if vertex in graph[v]:
            graph[v].remove(vertex)
    
    return graph