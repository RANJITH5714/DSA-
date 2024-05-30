vertexData = ['A', 'B', 'C', 'D']

adjacency_matrix = [
    [0, 1, 1, 1],  # Edges for A
    [1, 0, 1, 0],  # Edges for B
    [1, 1, 0, 0],  # Edges for C
    [1, 0, 0, 0]   # Edges for D
]

def print_adjacency_matrix(matrix):
    print("Adjacency Matrix:")
    for row in matrix:
        print(row)

def print_connections(matrix, vertices):
    print("\nConnections for each vertex:")
    for i in range(len(vertices)):
        print(f"{vertices[i]}: ", end="")
        for j in range(len(vertices)):
            if matrix[i][j]:  # if there is a connection
                print(vertices[j], end=" ")
        print()  # new line

print_adjacency_matrix(adjacency_matrix)
print_connections(adjacency_matrix, vertexData)
class Graph:
    def __init__(self, size):
        self.adj_matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.vertex_data = [''] * size
        
    def add_edge(self, u, v, weight):
        if 0 <= u < self.size and 0 <= v < self.size:
            self.adj_matrix[u][v] = weight
            #self.adj_matrix[v][u] = weight  # For undirected graph

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data
    def bellman_ford(self, start_vertex_data):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.size
        distances[start_vertex] = 0

        for i in range(self.size - 1):
            for u in range(self.size):
                for v in range(self.size):
                    if self.adj_matrix[u][v] != 0:
                        if distances[u] + self.adj_matrix[u][v] < distances[v]:
                            distances[v] = distances[u] + self.adj_matrix[u][v]
                            print(f"Relaxing edge {self.vertex_data[u]}-{self.vertex_data[v]}, Updated distance to {self.vertex_data[v]}: {distances[v]}")

        return distances
        class Graph:
    def __init__(self, size):
        self.adj_matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.vertex_data = [''] * size

    def add_edge(self, u, v, c):
        self.adj_matrix[u][v] = c

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data

    def dfs(self, s, t, visited=None, path=None):
        if visited is None:
            visited = [False] * self.size
        if path is None:
            path = []

        visited[s] = True
        path.append(s)

        if s == t:
            return path

        for ind, val in enumerate(self.adj_matrix[s]):
            if not visited[ind] and val > 0:
                result_path = self.dfs(ind, t, visited, path.copy())
                if result_path:
                    return result_path

        return None

    def fordFulkerson(self, source, sink):
        max_flow = 0

        path = self.dfs(source, sink)
        while path:
            path_flow = float("Inf")
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_flow = min(path_flow, self.adj_matrix[u][v])

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.adj_matrix[u][v] -= path_flow
                self.adj_matrix[v][u] += path_flow

            max_flow += path_flow

            path_names = [self.vertex_data[node] for node in path]
            print("Path:", " -> ".join(path_names), ", Flow:", path_flow)

            path = self.dfs(source, sink)

        return max_flow

g = Graph(6)
vertex_names = ['s', 'v1', 'v2', 'v3', 'v4', 't']
for i, name in enumerate(vertex_names):
    g.add_vertex_data(i, name)

g.add_edge(0, 1, 3)  # s  -> v1, cap: 3
g.add_edge(0, 2, 7)  # s  -> v2, cap: 7
g.add_edge(1, 3, 3)  # v1 -> v3, cap: 3
g.add_edge(1, 4, 4)  # v1 -> v4, cap: 4
g.add_edge(2, 1, 5)  # v2 -> v1, cap: 5
g.add_edge(2, 4, 3)  # v2 -> v4, cap: 3
g.add_edge(3, 4, 3)  # v3 -> v4, cap: 3
g.add_edge(3, 5, 2)  # v3 -> t,  cap: 2
g.add_edge(4, 5, 6)  # v4 -> t,  cap: 6

source = 0; sink = 5

print("The maximum possible flow is %d " % g.fordFulkerson(source, sink))
                    visited[ind] = True
                    parent[ind] = u
​
        return visited[t]
​
    def edmonds_karp(self, source, sink):
        parent = [-1] * self.size
        max_flow = 0
​
        while self.bfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.adj_matrix[parent[s]][s])
                s = parent[s]
​
            max_flow += path_flow
            v = sink
            while(v != source):
                u = parent[v]
                self.adj_matrix[u][v] -= path_flow
                self.adj_matrix[v][u] += path_flow
                v = parent[v]
​
            path = []
            v = sink
            while(v != source):
                path.append(v)
                v = parent[v]
            path.append(source)
            path.reverse()
            path_names = [self.vertex_data[node] for node in path]
            print("Path:", " -> ".join(path_names), ", Flow:", path_flow)
​
        return max_flow
​
# Example usage:
g = Graph(6)
vertex_names = ['s', 'v1', 'v2', 'v3', 'v4', 't']
for i, name in enumerate(vertex_names):
    g.add_vertex_data(i, name)
​
g.add_edge(0, 1, 3)  # s  -> v1, cap: 3
g.add_edge(0, 2, 7)  # s  -> v2, cap: 7
g.add_edge(1, 3, 3)  # v1 -> v3, cap: 3
g.add_edge(1, 4, 4)  # v1 -> v4, cap: 4
g.add_edge(2, 1, 5)  # v2 -> v1, cap: 5
g.add_edge(2, 4, 3)  # v2 -> v4, cap: 3
g.add_edge(3, 4, 3)  # v3 -> v4, cap: 3
g.add_edge(3, 5, 2)  # v3 -> t,  cap: 2
g.add_edge(4, 5, 6)  # v4 -> t,  cap: 6
​
source = 0; sink = 5
print("The maximum possible flow is %d " % g.edmonds_karp(source, sink))
def gcd_division(a, b):
    while b != 0:
        remainder = a % b
        print(f"{a} = {a//b} * {b} + {remainder}")
        a = b
        b = remainder
    return a

a = 120
b = 25
print("The Euclidean algorithm using division:\n")
print(f"The GCD of {a} and {b} is: {gcd_division(a, b)}")
def nearest_neighbor_tsp(distances):
    n = len(distances)
    visited = [False] * n
    route = [0]
    visited[0] = True
    total_distance = 0

    for _ in range(1, n):
        last = route[-1]
        nearest = None
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and distances[last][i] < min_dist:
                min_dist = distances[last][i]
                nearest = i
        route.append(nearest)
        visited[nearest] = True
        total_distance += min_dist

    total_distance += distances[route[-1]][0]
    route.append(0)
    return route, total_distance

distances = [
    [0, 2, 2, 5, 9, 3],
    [2, 0, 4, 6, 7, 8],
    [2, 4, 0, 8, 6, 3],
    [5, 6, 8, 0, 4, 9],
    [9, 7, 6, 4, 0, 10],
    [3, 8, 3, 9, 10, 0]
]

route, total_distance = nearest_neighbor_tsp(distances)
print("Route:", route)
print("Total distance:", total_distance)



  
        
