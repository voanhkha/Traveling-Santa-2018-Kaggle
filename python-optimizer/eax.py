import random

# Like a regular dict, except we can also get
# a random key in (amortized) constant time.
class Dict:

    # The keys and values are stored as lists. An index
    # associates the key to the position of the keys/values.
    def __init__(self):
        self.index = dict()
        self.keys = list()
        self.values = list()

    def __contains__(self, key):
        return key in self.index

    def __len__(self):
        return len(self.keys)

    # Inserts the new key/value at the end of the lists.
    def __setitem__(self, key, value):
        assert key not in self
        self.index[key] = len(self.keys)
        self.keys.append(key)
        self.values.append(value)

    def __getitem__(self, key):
        return self.values[self.index[key]]

    # Overwrites the key/value to be deleted by the last
    # key/value in the lists, then shrinks the lists.
    def __delitem__(self, key):
        assert key in self
        i = self.index[key]
        k = self.keys[-1]
        v = self.values[-1]
        self.index[k] = i
        self.keys[i] = k
        self.values[i] = v
        del self.index[key]
        del self.keys[-1]
        del self.values[-1]

    def random_key(self):
        return random.choice(self.keys)

# An undirected graph. Might contain redundant edges.
class Graph:

    # A dictionary associates each vertex
    # to the list of vertices it is adjacent to.
    def __init__(self):
        self.edges = Dict()

    def empty(self):
        return len(self.edges) == 0

    def add_edge(self, v0, v1):
        # Forward
        if v0 not in self.edges:
            self.edges[v0] = list()
        self.edges[v0].append(v1)
        # Backward
        if v1 not in self.edges:
            self.edges[v1] = list()
        self.edges[v1].append(v0)

    def remove_edge(self, v0, v1):
        # Forward
        self.edges[v0].remove(v1)
        if len(self.edges[v0]) == 0:
            del self.edges[v0]
        # Backward
        self.edges[v1].remove(v0)
        if len(self.edges[v1]) == 0:
            del self.edges[v1]

    # Returns a random vertex from the graph.
    def random_vertex(self):
        assert not self.empty()
        return self.edges.random_key()

    # Returns a random vertex adjacent to the one specified.
    def random_adjacent(self, v):
        assert v in self.edges
        return random.choice(self.edges[v])

    # Creates a graph from subtours.
    @staticmethod
    def from_subtours(subtours):
        graph = Graph()
        for subtour in subtours:
            assert subtour[0] == subtour[-1]
            for i in range(len(subtour) - 1):
                v0, v1 = subtour[i], subtour[i + 1]
                graph.add_edge(v0, v1)
        return graph

    # Creates subtours from a graph.
    def to_subtours(self, v=0):
        while not self.empty():
            # Prefer to start from v
            if v not in self.edges:
                v = self.random_vertex()
            subtour = [v]
            while True:
                v0 = subtour[-1]
                if v0 not in self.edges: break
                v1 = self.random_adjacent(v0)
                self.remove_edge(v0, v1)
                subtour.append(v1)
            assert subtour[0] == subtour[-1]
            yield subtour

# AB-cycles decomposition. Class used as namespace.
class AB_Cycles:

    # Checks for an even-length cycle at the end of the path.
    # Returns the index of its first vertex.
    @staticmethod
    def _find_cycle(path):
        assert len(path) > 0
        last_vertex = path[-1]
        for index in range(len(path) - 3, -1, -2):
            if path[index] == last_vertex:
                return index

    # Checks for an AB-cycle at the end of the path. Removes
    # it from the path. Returns the pruned path and the cycle.
    @staticmethod
    def _remove_cycle(path):
        index = AB_Cycles._find_cycle(path)
        if index is not None:
            path, cycle = path[:(index + 1)], path[index:]
            # Convert from BA-cycle to AB-cycle if required
            if len(path) & 1 == 0: cycle = cycle[1:] + [cycle[1]]
            # Clean up the path if it does not contain a full edge
            if len(path) == 1: path = []
        else: cycle = None
        return path, cycle

    # Decomposes 2 tours into AB-cycles. Creates a path alternately
    # picking edges from A and B until all the edges are used. Removes
    # the AB-cycles from the path when they appear.
    @staticmethod
    def ab_cycles(tour_a, tour_b):
        assert len(tour_a) == len(tour_b)
        path = list()
        graphs = (
            Graph.from_subtours([tour_a]),
            Graph.from_subtours([tour_b]))
        while True:
            # If the path is empty
            if len(path) == 0:
                # Terminate if appropriate
                if graphs[0].empty(): break
                # Add a first vertex otherwise
                vertex = graphs[0].random_vertex()
                path.append(vertex)
            # Alternately pick edges from A and B
            ab = 1 - (len(path) & 1)
            # Move an edge from A or B to the path
            v0 = path[-1]
            v1 = graphs[ab].random_adjacent(v0)
            graphs[ab].remove_edge(v0, v1)
            path.append(v1)
            # Remove and yield an eventual AB-cycle
            path, cycle = AB_Cycles._remove_cycle(path)
            if cycle is not None: yield cycle
        # Final checks
        assert graphs[0].empty()
        assert graphs[1].empty()
        assert len(path) == 0

# EAX child construction. Class used as namespace.
class EAX:

    # E-set for EAX(rand).
    @staticmethod
    def _eset_rand(ab_cycles):
        for ab_cycle in ab_cycles:
            if len(ab_cycle) > 3 and random.uniform(0, 1) > 0.5:
                yield ab_cycle

    # Generates the intermediate solution. Takes A, removes
    # the E-set edges coming from A, adds those coming from B.
    # Returns a graph of the resulting disjoint subtours.
    @staticmethod
    def _intermediate_solution(tour_a, eset):
        # Create a graph from A
        graph = Graph.from_subtours([tour_a])
        # Add the E-set edges from A, remove those from B
        for ab_cycle in eset:
            ab = 0
            for i in range(len(ab_cycle) - 1):
                v0, v1 = ab_cycle[i], ab_cycle[i + 1]
                if ab == 0: graph.remove_edge(v0, v1)
                else: graph.add_edge(v0, v1)
                ab = 1 - ab
        return graph

    # Separates the shortest subtour from the others.
    @staticmethod
    def _shortest_subtour(subtours):
        lengths = [len(subtour) for subtour in subtours]
        i = lengths.index(min(lengths))
        shortest = subtours[i]
        others = subtours[:i] + subtours[(i + 1):]
        return shortest, others

    # Finds quads v0, v1, v2, v3 such that (v0, v1) is an edge
    # of the shortest subtour, (v2, v3) is an edge of any other
    # subtour, v2 ∈ N(v0), and v3 ∈ N(v0).
    @staticmethod
    def _quads(subtours, neighbors):
        # Separate the shortest subtour
        shortest, others = EAX._shortest_subtour(subtours)
        # List of edges from the shortest subtour
        x = list()
        for i in range(len(shortest) - 1):
            v0, v1 = shortest[i], shortest[i + 1]
            x.append((v0, v1))
        # Set of edges from the other subtours
        y = set()
        for subtour in others:
            for i in range(len(subtour) - 1):
                v0, v1 = subtour[i], subtour[i + 1]
                y.add((v0, v1))
                y.add((v1, v0))
        # Find quads
        for v0, v1 in x:
            for v2 in neighbors[v0]:
                for v3 in neighbors[v1]:
                    if (v2, v3) in y:
                        yield v0, v1, v2, v3

    # Finds the quad v0, v1, v2, v3 which merges the shortest subtour
    # and another while minimizing the total distance when we remove
    # (v0, v1) and (v2, v3) and add (v0, v2) and (v1, v3).
    @staticmethod
    def _min_quad(subtours, world, neighbors):
        min_distance = None
        min_quad = None
        for v0, v1, v2, v3 in EAX._quads(subtours, neighbors):
            removed = world.distance(v0, v1) + world.distance(v2, v3)
            for (v2, v3) in ((v2, v3), (v3, v2)):
                added = world.distance(v0, v2) + world.distance(v1, v3)
                distance = added - removed
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_quad = v0, v1, v2, v3
        return min_quad

    # Generates the offspring solution from the intermediate one.
    # Greedily merges the subtours until only one remains.
    def _offspring_solution(graph, world, neighbors):
        while True:
            subtours = tuple(graph.to_subtours())
            if len(subtours) == 1: return subtours[0]
            print("subtours", len(subtours))
            v0, v1, v2, v3 = EAX._min_quad(subtours, world, neighbors)
            graph = Graph.from_subtours(subtours)
            graph.remove_edge(v0, v1)
            graph.remove_edge(v2, v3)
            graph.add_edge(v0, v2)
            graph.add_edge(v1, v3)

    # Generates the EAX offspring. Pieces everything together.
    @staticmethod
    def eax(tour_a, tour_b, world, neighbors):
        ab_cycles = AB_Cycles.ab_cycles(tour_a, tour_b)
        eset = EAX._eset_rand(ab_cycles)
        intermediate = EAX._intermediate_solution(tour_a, eset)
        return EAX._offspring_solution(intermediate, world, neighbors)


if __name__ == "__main__":

    # Simple tests for Dict.

    d = Dict()
    d[7] = "abc"
    assert d[7] == "abc"
    d[12] = "def"
    assert d[12] == "def"
    assert len(d) == 2
    assert 7 in d
    assert 12 in d
    assert 123 not in d
    del d[7]
    assert len(d) == 1
    assert 7 not in d
    assert 12 in d
    assert d.random_key() == 12
    assert d[12] == "def"
    del d

    # Simple tests for Graph.

    g = Graph.from_subtours([[0, 1, 2, 3, 0]])
    assert not g.empty()
    assert g.random_vertex() in [0, 1, 2, 3]
    assert g.random_adjacent(0) in [1, 3]
    g.remove_edge(0, 1)
    g.remove_edge(1, 2)
    g.remove_edge(3, 0)
    assert g.random_vertex() in [2, 3]
    assert g.random_adjacent(2) == 3
    g.remove_edge(2, 3)
    assert g.empty()
    del g

    # Simple tests for AB_Cycles.

    n = 100
    tour1 = list(range(1, n))
    random.shuffle(tour1)
    tour1 = [0] + tour1 + [0]
    tour2 = list(range(1, n))
    random.shuffle(tour2)
    tour2 = [0] + tour2 + [0]
    counter = 0
    for cycle in AB_Cycles.ab_cycles(tour1, tour2):
        assert cycle[0] == cycle[-1]
        counter += len(cycle) - 1
    assert counter == 2 * n
    del tour1, tour2, counter

    # Real-world test for EAX.

    import tsp

    world = tsp.World("resources/cities.csv")
    popmusic = tsp.Neighbors.load_pickle("resources/popmusic.pickle")

    a = tsp.Tour.load_csv("raw_tours/raw1502605.csv")
    b = tsp.Tour.load_csv("raw_tours/raw1502650.csv")
    c = EAX.eax(a, b, world, popmusic)

    print("A", world.score(a, 0))
    print("B", world.score(b, 0))
    print("C", world.score(c, 0))
    print("distance(A, B)", tsp.Tour.distance(a, b))
    print("distance(C, A)", tsp.Tour.distance(c, a))
    print("distance(C, B)", tsp.Tour.distance(c, b))

    tsp.Tour.save_csv(c, 'merged_tour.csv')