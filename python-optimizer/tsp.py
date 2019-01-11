import math
import pickle as p

class World:

    def __init__(self, filename):
        cities_x, cities_y, n = World._read_cities(filename)
        self.cities_x = tuple(cities_x)
        self.cities_y = tuple(cities_y)
        self.n = n
        primes = World._prime_sieve(self.n)
        self.non_primes = tuple(1 - p for p in primes)

    # Reads cities from CSV file.
    @staticmethod
    def _read_cities(filename):
        cities_x, cities_y = [], []
        with open(filename) as s:
            for i, line in enumerate(s.readlines()):
                if i == 0: continue
                _, x, y = line.strip().split(',')
                cities_x.append(float(x))
                cities_y.append(float(y))
        assert len(cities_x) == i and len(cities_y) == i
        return cities_x, cities_y, i

    # Sieve of Eratosthenes.
    @staticmethod
    def _prime_sieve(n):
        assert n >= 2
        primes = [False, False] + [True] * (n - 2)
        for i in range(2, n):
            if primes[i]:
                for j in range(2 * i, n, i):
                    primes[j] = False
        return primes

    # Distance between 2 cities.
    def distance(self, i, j):
        x = self.cities_x[i] - self.cities_x[j]
        y = self.cities_y[i] - self.cities_y[j]
        return math.sqrt((x * x) + (y * y))

    # Tour score.
    def score(self, tour, percent=0.1):
        distances = []
        for i in range(1, len(tour)):
            distances.append(self.distance(tour[i], tour[i-1]))
        penalties = []
        for i in range(9, len(tour) - 1, 10):
            penalties.append(self.non_primes[tour[i]] * distances[i])
        return sum(distances) + percent * sum(penalties)

# Class used as namespace.
class Tour:

    # Reads CSV submission.
    @staticmethod
    def load_csv(filename):
        tour = []
        with open(filename) as stream:
            for i, line in enumerate(stream.readlines()):
                if i == 0: continue
                city = int(line)
                tour.append(city)
        assert (tour[0] == 0) and (tour[-1] == 0)
        return tour

    # Writes CSV submission.
    @staticmethod
    def save_csv(tour, filename):
        with open(filename, "w") as s:
            s.write("Path\n")
            for city in tour:
                s.write("%d\n" % city)

    # Number of non-common edges between 2 tours.
    @staticmethod
    def edge_distance(tour1, tour2):
        assert len(tour1) == len(tour2)
        def edges(tour):
            for i in range(len(tour) - 1):
                yield tuple(sorted((tour[i], tour[i + 1])))
        edges2 = set(edges(tour2))
        distance = 0
        for edge in edges(tour1):
            if edge not in edges2:
                distance += 1
        return distance

    # Number of non-common placed primes between 2 tours.
    def prime_distance(tour1, tour2):
        assert len(tour1) == len(tour2)
        primes = World._prime_sieve(len(tour1))
        def placed_primes(tour):
            for i in range(9, len(tour) - 1, 10):
                if primes[tour[i]]: yield tour[i]
        primes1 = set(placed_primes(tour1))
        primes2 = set(placed_primes(tour2))
        return len(primes1 | primes2) - len(primes1 & primes2)

    @staticmethod
    def distance(tour1, tour2):
        return (
            Tour.edge_distance(tour1, tour2),
            Tour.prime_distance(tour1, tour2))

# Class used as namespace.
class Neighbors:

    # Loads pickled neighbors list.
    @staticmethod
    def load_pickle(filename):
        with open(filename, "rb") as stream:
            return p.load(stream)
