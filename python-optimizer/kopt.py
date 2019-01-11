import random
import math

cities_x, cities_y = [], []
with open("resources/cities.csv") as s:
    for i, line in enumerate(s.readlines()):
        if i == 0: continue
        _, x, y = line.strip().split(',')
        cities_x.append(float(x))
        cities_y.append(float(y))

def shuffle(l):
    l = list(l)
    random.shuffle(l)
    return iter(l)

# Finds candidates for k-opt moves. Class used as namepace.
class Candidates:

    # Precomputation to find the position
    # of a city in the tour in constant time.
    @staticmethod
    def tour_inverse(tour, start, stop):
        tour_inverse = [-1] * len(tour)
        for i in range(start, stop - 1):
            tour_inverse[tour[i]] = i
        return tour_inverse

    # Finds pairs a, b such that a < b and b ∈ N(a) and b+1 ∈ N(a+1).
    @staticmethod
    def pairs_2opt(tour, neighbors, start, stop):
        tour_inverse = Candidates.tour_inverse(tour, start, stop)
        def n(x):
            for y in neighbors[tour[x]]:
                z = tour_inverse[y]
                if z != -1: yield z
        for a0 in shuffle(range(start, stop - 1)):
            a1 = a0 + 1
            for b0 in n(a0):
                b1 = b0 + 1
                if a0 < b0 and b1 in n(a1):
                    yield a0, b0

    # Finds triples a, b, c such that a < b < c and
    # b+1 ∈ N(a) and c+1 ∈ N(b) and a+1 ∈ N(c).
    @staticmethod
    def triples_3opt(tour, neighbors, start, stop):
        tour_inverse = Candidates.tour_inverse(tour, start, stop)
        def n(x):
            for y in neighbors[tour[x]]:
                z = tour_inverse[y]
                if z != -1: yield z
        for a0 in shuffle(range(start, stop - 1)):
            a1 = a0 + 1
            for b1 in n(a0):
                b0 = b1 - 1
                if a0 < b0:
                    for c1 in n(b0):
                        c0 = c1 - 1
                        if b0 < c0 and a1 in n(c0):
                            yield a0, b0, c0

    # Finds quads a, b, c, d such that a < b < c < d and
    # c+1 ∈ N(a) and c ∈ N(a+1) and and d+1 ∈ N(b) and d ∈ N(b+1).
    @staticmethod
    def quads_4opt(tour, neighbors, start, stop):
        tour_inverse = Candidates.tour_inverse(tour, start, stop)
        def n(x):
            for y in neighbors[tour[x]]:
                z = tour_inverse[y]
                if z != -1: yield z
        pairs = []
        for a0 in range(start, stop - 1):
            a1 = a0 + 1
            for b1 in n(a0):
                b0 = b1 - 1
                if a0 < b0 and b0 in n(a1):
                    pairs.append((a0, b0))
        for i in shuffle(range(len(pairs))):
            a, c = pairs[i]
            quads = []
            for j in range(i + 1, len(pairs)):
                b, d = pairs[j]
                if not a < b: continue
                if not b < c: break
                if c < d: quads.append((a, b, c, d))
            for quad in shuffle(quads):
                yield quad

    @staticmethod
    def five_opt_candidates(tour, nearest_neighbors, start, stop):
        for c, tourOrder in LK_5Move(tour, nearest_neighbors, start, stop):
            if not allUnique(c): continue
            tour_inverse = [0] * len(tour)
            for i, j in enumerate(tour):
                tour_inverse[j] = i
            tourPos = [tour_inverse[j] for j in c]
            yield c, tourOrder, tourPos

    @staticmethod
    def four_opt_candidates(tour, nearest_neighbors, start, stop):
        for c, tourOrder, valid_4opt_flag, G4a in LK_4Move(tour, nearest_neighbors, start, stop):
            if valid_4opt_flag: 
               if not allUnique(c): continue
               tour_inverse = [0] * len(tour)
               for i, j in enumerate(tour):
                  tour_inverse[j] = i
               tourPos = [tour_inverse[j] for j in c]
               yield c, tourOrder, tourPos    

# Abstract class defining the segment operations for the
# k-opt moves. Implementations: ScoreBuilder and TourBuilder.
class Builder:

    def noop(self):
        self.add_forward(0, -1)
        return self

    # 2-opt move.
    # From  0 .. a a+1 ..  b  b+1 .. 0
    # To    0 .. a  b  .. a+1 b+1 .. 0

    def forward_2opt(self, a, b):
        self.add_forward(0, a)
        self.add_backward(a + 1, b)
        self.add_forward(b + 1, -1)
        return self

    def backward_2opt(self, a, b):
        self.add_backward(b + 1, -1)
        self.add_forward(a + 1, b)
        self.add_backward(0, a)
        return self

    # 3-opt move (3x 2-opt moves).
    # From 0 .. a a+1 .. b b+1 .. c c+1 .. 0
    # To   0 .. a b+1 .. c a+1 .. b c+1 .. 0

    def forward_3opt(self, a, b, c):
        self.add_forward(0, a)
        self.add_forward(b + 1, c)
        self.add_forward(a + 1, b)
        self.add_forward(c + 1, -1)
        return self

    def backward_3opt(self, a, b, c):
        self.add_backward(c + 1, -1)
        self.add_backward(a + 1, b)
        self.add_backward(b + 1, c)
        self.add_backward(0, a)
        return self

    # 4-opt move (double bridge move).
    # From  0 .. a a+1 .. b b+1 .. c c+1 .. d d+1 .. 0
    # To    0 .. a c+1 .. d b+1 .. c a+1 .. b d+1 .. 0

    def forward_4opt(self, a, b, c, d):
        self.add_forward(0, a)
        self.add_forward(c + 1, d)
        self.add_forward(b + 1, c)
        self.add_forward(a + 1, b)
        self.add_forward(d + 1, -1)
        return self

    def backward_4opt(self, a, b, c, d):
        self.add_backward(d + 1, -1)
        self.add_backward(a + 1, b)
        self.add_backward(b + 1, c)
        self.add_backward(c + 1, d)
        self.add_backward(0, a)
        return self

    def forward_5opt_sequential(self, tencities, tourOrder, tourPosition):
        ## c = [c1, c2, ..., c9, cA]: city name, not position index! 
        ## tourOrder = '129A347856': cyclic order of [c1, c2..., cA]
        ## tourPosition = [p1,p2,...,pA]: position index of [c1,c2,...,cA]
        # Step 1: Search for city with minimum index (nearest to city 0)
        #print(tencities)
        #print(tourOrder)
        ten_order = parse_tourOrder_5opt(tencities, tourOrder)
        ind_1 = min(tourPosition)
        city_ind_1 = tencities[tourPosition.index(ind_1)]
        self.add_forward(0, ind_1)
        # Step 2: Get neighbor of city_ind_1, that is right next to c_i if i odd, or left next to c_i if i even
        for count in range(4):
            city_ind_2 = get_nn_quintuple(city_ind_1, tencities)
            ind_2 = tourPosition[tencities.index(city_ind_2)]
            city_ind_3, fw = get_segment_quintuple(city_ind_2, ten_order)
            ind_3  = tourPosition[tencities.index(city_ind_3)]
            if ind_2 < ind_3: self.add_forward(ind_2, ind_3)
            else: self.add_backward(ind_3, ind_2)
            # update for next loop
            city_ind_1 = city_ind_3
        # close the tour, finish
        city_ind_2 = get_nn_quintuple(city_ind_1, tencities)
        ind_2 = tourPosition[tencities.index(city_ind_2)]
        self.add_forward(ind_2, -1)
        return self


    def backward_5opt_sequential(self, tencities, tourOrder, tourPosition):
        ## c = [c1, c2, ..., c9, cA]: city name, not position index! 
        ## tourOrder = '129A347856': cyclic order of [c1, c2..., cA]
        ## tourPosition = [p1,p2,...,pA]: position index of [c1,c2,...,cA]
        # Step 1: Search for city with maximum index (nearest to city 0)
        ten_order = parse_tourOrder_5opt(tencities, tourOrder)
        ind_1 = max(tourPosition)
        city_ind_1 = tencities[tourPosition.index(ind_1)]
        self.add_backward(ind_1, -1)
        # Step 2: Get neighbor of city_ind_1, that is right next to c_i if i odd, or left next to c_i if i even
        for count in range(4):
            city_ind_2 = get_nn_quintuple(city_ind_1, tencities)
            ind_2 = tourPosition[tencities.index(city_ind_2)]
            city_ind_3, fw = get_segment_quintuple(city_ind_2, ten_order)
            ind_3  = tourPosition[tencities.index(city_ind_3)]
            if ind_2 < ind_3: self.add_backward(ind_2, ind_3)
            else: self.add_forward(ind_3, ind_2)

            # update for next loop
            city_ind_1 = city_ind_3

        # close the tour, finish
        city_ind_2 = get_nn_quintuple(city_ind_1, tencities)
        ind_2 = tourPosition[tencities.index(city_ind_2)]
        self.add_backward(0, ind_2)
        return self


    def forward_4opt_sequential(self, eightcities, tourOrder, tourPosition):
        eight_order = parse_tourOrder_4opt(eightcities, tourOrder)
        ind_1 = min(tourPosition)
        city_ind_1 = eightcities[tourPosition.index(ind_1)]
        self.add_forward(0, ind_1)
        for count in range(3):
            city_ind_2 = get_nn_quadtuple(city_ind_1, eightcities)
            ind_2 = tourPosition[eightcities.index(city_ind_2)]
            city_ind_3, fw = get_segment_quadtuple(city_ind_2, eight_order)
            ind_3  = tourPosition[eightcities.index(city_ind_3)]
            if ind_2 < ind_3: self.add_forward(ind_2, ind_3)
            else: self.add_backward(ind_3, ind_2)
            # update for next loop
            city_ind_1 = city_ind_3
        # close the tour, finish
        city_ind_2 = get_nn_quadtuple(city_ind_1, eightcities)
        ind_2 = tourPosition[eightcities.index(city_ind_2)]
        self.add_forward(ind_2, -1)
        return self


    def backward_4opt_sequential(self, eightcities, tourOrder, tourPosition):
        eight_order = parse_tourOrder_4opt(eightcities, tourOrder)
        ind_1 = max(tourPosition)
        city_ind_1 = eightcities[tourPosition.index(ind_1)]
        self.add_backward(ind_1, -1)
        for count in range(3):
            city_ind_2 = get_nn_quadtuple(city_ind_1, eightcities)
            ind_2 = tourPosition[eightcities.index(city_ind_2)]
            city_ind_3, fw = get_segment_quadtuple(city_ind_2, eight_order)
            ind_3  = tourPosition[eightcities.index(city_ind_3)]
            if ind_2 < ind_3: self.add_backward(ind_2, ind_3)
            else: self.add_forward(ind_3, ind_2)
            # update for next loop
            city_ind_1 = city_ind_3
        # close the tour, finish
        city_ind_2 = get_nn_quadtuple(city_ind_1, eightcities)
        ind_2 = tourPosition[eightcities.index(city_ind_2)]
        self.add_backward(0, ind_2)
        return self

def parse_tourOrder_5opt(tencities, tourOrder):
    # Example: tencities = [10,20,30,40,50,60,70,80,90,100], tourOrder =  = '124356A987' 
    #            -> output=[10,20,40,30,50,60,100,90,80,70]
    tenorder = []
    for char in tourOrder[3:]:
        if char=='A': tenorder.append(tencities[9])
        else: tenorder.append(tencities[int(char)-1])
    return tenorder

def parse_tourOrder_4opt(eightcities, tourOrder):
    eightorder = []
    for char in tourOrder[3:]:
        eightorder.append(eightcities[int(char)-1])
    return eightorder


def get_nn_quintuple(city, tencities):
    # if city = c1, output = cA (vice versa), or if city = c6, output = c7...
    ind_city = tencities.index(city)
    if ind_city == 0: return tencities[9]
    if ind_city == 1: return tencities[2]
    if ind_city == 2: return tencities[1]
    if ind_city == 3: return tencities[4]
    if ind_city == 4: return tencities[3]
    if ind_city == 5: return tencities[6]
    if ind_city == 6: return tencities[5]
    if ind_city == 7: return tencities[8]
    if ind_city == 8: return tencities[7]
    if ind_city == 9: return tencities[0]

def get_nn_quadtuple(city, eightcities):
    # if city = c1, output = cA (vice versa), or if city = c6, output = c7...
    ind_city = eightcities.index(city)
    if ind_city == 0: return eightcities[7]
    if ind_city == 1: return eightcities[2]
    if ind_city == 2: return eightcities[1]
    if ind_city == 3: return eightcities[4]
    if ind_city == 4: return eightcities[3]
    if ind_city == 5: return eightcities[6]
    if ind_city == 6: return eightcities[5]
    if ind_city == 7: return eightcities[0]

def get_segment_quintuple(city, tenorder): # return city of the same segment
    ind_city = tenorder.index(city)
    if ind_city == 0: return tenorder[9], 0
    if ind_city == 1: return tenorder[2], 1
    if ind_city == 2: return tenorder[1], 0
    if ind_city == 3: return tenorder[4], 1
    if ind_city == 4: return tenorder[3], 0
    if ind_city == 5: return tenorder[6], 1
    if ind_city == 6: return tenorder[5], 0
    if ind_city == 7: return tenorder[8], 1
    if ind_city == 8: return tenorder[7], 0
    if ind_city == 9: return tenorder[0], 1

def get_segment_quadtuple(city, eightorder): # return city of the same segment
    ind_city = eightorder.index(city)
    if ind_city == 0: return eightorder[7], 0
    if ind_city == 1: return eightorder[2], 1
    if ind_city == 2: return eightorder[1], 0
    if ind_city == 3: return eightorder[4], 1
    if ind_city == 4: return eightorder[3], 0
    if ind_city == 5: return eightorder[6], 1
    if ind_city == 6: return eightorder[5], 0
    if ind_city == 7: return eightorder[0], 1

def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

# Precomputations to score k-opt moves faster.
class Scorer:

    def __init__(self, tour, world, percent=0.1):
        self.tour = tour
        self.world = world
        self.percent = percent
        self.forward_scores = self._preprocess(tour)
        self.backward_scores = self._preprocess(tour[::-1])

    # Cumulative sum of the scores at each city for the 10 penaly offsets.
    def _preprocess(self, tour):
        distances = []
        for i in range(1, len(tour)):
            distance = self.world.distance(tour[i], tour[i - 1])
            distances.append(distance)
        scores = []
        for j in range(10):
            penalties = [0] * len(distances)
            for k in range(j, len(distances), 10):
                p = self.world.non_primes[tour[k]] * distances[k]
                penalties[k] = self.percent * p
            cumsum = []
            cumsum.append(distances[0] + penalties[0])
            for k in range(1, len(distances)):
                cumsum.append(cumsum[-1] + distances[k] + penalties[k])
            scores.append(cumsum)
        return scores

    # Scores a segment between indices a and b included
    # starting at a new index and in the original direction.
    def score_forward(self, a, b, new_start):
        assert b >= a
        offset = (a - new_start + 9) % 10
        distance = self.forward_scores[offset][b - 1]
        if a > 0: distance -= self.forward_scores[offset][a - 1]
        return distance

    # Scores a segment between indices a and b included
    # starting at a new index and in the reverse direction.
    def score_backward(self, a, b, new_start):
        assert b >= a
        inv = lambda i: len(self.tour) - 1 - i
        offset = (inv(b) - new_start + 9) % 10
        distance = self.backward_scores[offset][inv(a) - 1]
        if inv(b) > 0: distance -= self.backward_scores[offset][inv(b) - 1]
        return distance

    # Scores an edge between cities i and j not necessarily in the tour.
    def score_edge(self, i, j, edge_index):
        assert i != j
        distance = self.world.distance(i, j)
        if edge_index % 10 == 9 and self.world.non_primes[i]:
            distance *= (1 + self.percent)
        return distance

# Scores a modified tour without creating it.
class ScoreBuilder(Builder):

    def __init__(self, scorer):
        self.scorer = scorer
        self.score = 0.0
        self.tour_length = 0
        self.last_city = None

    # Adds a segment in the original direction.
    def add_forward(self, a, b):
        if b == -1: b = len(self.scorer.tour) - 1
        assert b >= a
        # Add an edge between the last city of the tour
        # and the first city of the new segment
        if self.tour_length > 0:
            self.score += self.scorer.score_edge(
                self.last_city,
                self.scorer.tour[a],
                self.tour_length - 1)
        # Add the new segment
        if b > a:
            self.score += self.scorer.score_forward(
                a, b, self.tour_length)
        # Maintain the tour info
        self.tour_length += b - a + 1
        self.last_city = self.scorer.tour[b]

    # Adds a segment in the reverse direction.
    def add_backward(self, a, b):
        if b == -1: b = len(self.scorer.tour) - 1
        assert b >= a
        # Add an edge between the last city of the tour
        # and the first city of the new segment
        if self.tour_length > 0:
            self.score += self.scorer.score_edge(
                self.last_city,
                self.scorer.tour[b],
                self.tour_length - 1)
        # Add the new segment
        if b > a:
            self.score += self.scorer.score_backward(
                a, b, self.tour_length)
        # Maintain the tour info
        self.tour_length += b - a + 1
        self.last_city = self.scorer.tour[a]

    def out(self):
        return self.score

# Creates a modified tour.
class TourBuilder(Builder):

    def __init__(self, tour):
        self.tour = tour
        self.new_tour = []

    # Adds a segment in the original direction.
    def add_forward(self, a, b):
        if b == -1: b = len(self.tour) - 1
        assert b >= a
        self.new_tour.extend(self.tour[a:b+1])

    # Adds a segment in the reverse direction.
    def add_backward(self, a, b):
        if b == -1: b = len(self.tour) - 1
        assert b >= a
        self.new_tour.extend(self.tour[a:b+1][::-1])

    def out(self):
        return self.new_tour

# Move abstraction for convenience.
class Move:

    def __init__(
            self, name, candidates,
            score_forward, score_backward,
            tour_forward, tour_backward, limit):
        self.name = name
        self.candidates = candidates
        self.score_forward = score_forward
        self.score_backward = score_backward
        self.tour_forward = tour_forward
        self.tour_backward = tour_backward
        self.limit = limit

    @staticmethod
    def move_2opt(neighbors):
        return Move(
            name="2-opt",
            candidates=lambda t, i, j: Candidates.pairs_2opt(t, neighbors, i, j),
            score_forward=lambda s, c: ScoreBuilder(s).forward_2opt(*c).out(),
            score_backward=lambda s, c: ScoreBuilder(s).backward_2opt(*c).out(),
            tour_forward=lambda t, c: TourBuilder(t).forward_2opt(*c).out(),
            tour_backward=lambda t, c: TourBuilder(t).backward_2opt(*c).out(),
            limit = None
            )

    @staticmethod
    def move_3opt(neighbors):
        return Move(
            name="3-opt (3x 2-opt)",
            candidates=lambda t, i, j: Candidates.triples_3opt(t, neighbors, i, j),
            score_forward=lambda s, c: ScoreBuilder(s).forward_3opt(*c).out(),
            score_backward=lambda s, c: ScoreBuilder(s).backward_3opt(*c).out(),
            tour_forward=lambda t, c: TourBuilder(t).forward_3opt(*c).out(),
            tour_backward=lambda t, c: TourBuilder(t).backward_3opt(*c).out(),
            limit = None
            )

    @staticmethod
    def move_4opt(neighbors):
        return Move(
            name="4-opt (double bridge)",
            candidates=lambda t, i, j: Candidates.quads_4opt(t, neighbors, i, j),
            score_forward=lambda s, c: ScoreBuilder(s).forward_4opt(*c).out(),
            score_backward=lambda s, c: ScoreBuilder(s).backward_4opt(*c).out(),
            tour_forward=lambda t, c: TourBuilder(t).forward_4opt(*c).out(),
            tour_backward=lambda t, c: TourBuilder(t).backward_4opt(*c).out(),
            limit = None
            )


    @staticmethod
    def move_4opt_sequential(neighbors):
        return Move(
            name="4-opt sequential",
            candidates=lambda t, i, j: Candidates.four_opt_candidates(t, neighbors, i, j),
            score_forward=lambda s, c: ScoreBuilder(s).forward_4opt_sequential(*c).out(),
            score_backward=lambda s, c: ScoreBuilder(s).backward_4opt_sequential(*c).out(),
            tour_forward=lambda t, c: TourBuilder(t).forward_4opt_sequential(*c).out(),
            tour_backward=lambda t, c: TourBuilder(t).backward_4opt_sequential(*c).out(),
            limit = 50000
            )

    @staticmethod
    def move_5opt_sequential(neighbors):
        return Move(
            name="5-opt sequential",
            candidates=lambda t, i, j: Candidates.five_opt_candidates(t, neighbors, i, j),
            score_forward=lambda s, c: ScoreBuilder(s).forward_5opt_sequential(*c).out(),
            score_backward=lambda s, c: ScoreBuilder(s).backward_5opt_sequential(*c).out(),
            tour_forward=lambda t, c: TourBuilder(t).forward_5opt_sequential(*c).out(),
            tour_backward=lambda t, c: TourBuilder(t).backward_5opt_sequential(*c).out(),
            limit = 50000
            )

# Class used as namespace.
class Optimizer:

    # Returns a tour after applying a single improving move,
    # or None if there is no such move.
    @staticmethod
    def optimize_one(
            tour, world, move, percent=0.1,
            min_gain=1e-3, start=None, stop=None):
        if start is None: start = 0
        if stop is None: stop = len(tour)
        scorer = Scorer(tour, world, percent)
        score = ScoreBuilder(scorer).noop().out()
        
        count = 0
        max_score = score - min_gain
        for candidate in move.candidates(tour, start, stop):
            if move.limit is not None:
                count += 1
                if count > move.limit: return None
            score = move.score_forward(scorer, candidate)
            if score < max_score:
                print(score)
                new_tour = move.tour_forward(tour, candidate)
                if len(new_tour) == 197770: return new_tour # because 4-opt and 5-opt sequential have a bug that may cause the bad tourlength


    # Returns a tour after applying all the improving moves,
    @staticmethod
    def optimize_move(
            tour, world, move, percent=0.1,
            min_gain=1e-3, start=None, stop=None):
        if start is None: start = 0
        if stop is None: stop = len(tour)
        #print(move.name, "[%d:%d]" % (start, stop))
        while True:
            new_tour = Optimizer.optimize_one(
                tour, world, move, percent=percent,
                min_gain=min_gain, start=start, stop=stop)
            if new_tour is None:
                return tour
            tour = new_tour

    # Returns a tour after applying all the improving moves.
    # Successively applies different types of moves.
    @staticmethod
    def optimize_moves(
            tour, world, moves, percent=0.1,
            min_gain=1e-3, start=None, stop=None):
        if start is None: start = 0
        if stop is None: stop = len(tour)
        stall_counter = -1
        while True:
            for move in moves:
                new_tour = Optimizer.optimize_move(
                    tour, world, move, percent=percent,
                    min_gain=min_gain, start=start, stop=stop)
                if new_tour == tour: stall_counter += 1
                else: stall_counter = 0
                tour = new_tour
                if stall_counter == len(moves) - 1:
                    return tour

#########################################################
#############   5-OPT LIBRARY    ######################
def t_pred(city, tour): ## returns the predecessor city for given position 
    return tour[(N + tour.index(city) -1) % N] 

def t_succ(city, tour): ## returns the successor city for given city 
    return tour[(tour.index(city)+1) % N] 

def Between(city_A, city_X, city_C, tour):
  ## Returns true if `x` is between `a` and `c` in tour
  ## with established direction (ascending position numbers)
  ## That is: when one begins a forward traversal of tour
  ## at city `a', then city `x` is reached before city `c'.
  ## Returns true if and only if:
  ##   a <= x <= c  or  c < a <= x  or  x <= c < a
  pA = tour.index(city_A)
  pX = tour.index(city_X)
  pC = tour.index(city_C)
  if pA <= pC:
    return (pX >= pA) and (pX <= pC)
  else:
    return (pX >= pA) or  (pX <= pC)


def inOrder(city_A, city_B, city_C, fwd, tour):
    if fwd:
        return Between(city_A, city_B, city_C, tour)
    else:
        return Between(city_C, city_B, city_A, tour)

def distance(i, j):
    x = cities_x[i] - cities_x[j]
    y = cities_y[i] - cities_y[j]
    return math.sqrt((x * x) + (y * y))

def LK_1Move(tour, nearest_neighbors, start, stop):
    for i in shuffle(range(start+1, stop-1)):
        #if i % 100 == 0: print(i)
        c1 = tour[i]
        c1_succ = t_succ(c1,tour)
        c1_pred = t_pred(c1,tour)
        for c2 in [c1_succ, c1_pred]:
            G1a = distance(c1, c2)
            yield c1, c2, G1a

def LK_2Move(tour,  nearest_neighbors, start, stop):
    #http://tsp-basics.blogspot.com/2017/06/lin-kernighan-algorithm-basics-part-2.html
    nb_nn_2opt = 5
    for c1, c2, G1a in LK_1Move(tour, nearest_neighbors, start, stop):
        fwd = (c2 == t_succ(c1, tour))
        c2_succ = t_succ(c2,tour)
        c2_pred = t_pred(c2,tour)
        for c3 in nearest_neighbors[c2][:nb_nn_2opt]: 
            if (c3 == c2_succ) or (c3 == c2_pred): continue
            G1 = G1a - distance(c2, c3)
            if G1 <= 0: break
            c3_succ = t_succ(c3, tour)
            c3_pred = t_pred(c3, tour)          
            for c4 in [c3_pred, c3_succ]:
                G2a = G1 + distance(c3, c4)
                if (fwd and (c4 == c3_succ))  or (not fwd and (c4 == c3_pred)):
                    yield (c1, c2, c3, c4), TO_1234, G2a
                else:
                    yield (c1, c2, c3, c4), TO_1243, G2a


def LK_3Move(tour,  nearest_neighbors, start, stop):
#http://tsp-basics.blogspot.com/2017/06/lin-kernighan-algorithm-basics-part-3.html
    nb_nn_3opt = 5
    for (c1, c2, c3, c4), tourOrderPrev, G2a in LK_2Move(tour, nearest_neighbors, start, stop):
        fwd = (c2 == t_succ(c1, tour))
        c4_succ = t_succ(c4, tour)
        c4_pred = t_pred(c4, tour)
        for c5 in nearest_neighbors[c4][:nb_nn_3opt]:
            if (c5 == c4_succ) or (c5 == c4_pred): continue
            G2 = G2a - distance(c4, c5)
            if G2  <= 0: break
            c5_succ = t_succ(c5, tour)
            c5_pred = t_pred(c5, tour)
            for c6 in [c5_succ, c5_pred]:
                G3a = G2 + distance(c5, c6)
                if tourOrderPrev == TO_1234:
                    if inOrder(c4, c5, c1, fwd, tour):
                        if (c6 == c2):  continue
                        if inOrder(c4, c5, c6, fwd, tour): yield (c1, c2, c3, c4, c5, c6), TO_123456, G3a  # disconnecting, but starts 5-opt
                        else: yield (c1, c2, c3, c4, c5, c6), TO_123465, G3a  # disconnecting, but starts 4-opt
                    else:
                        if (c6 == c1): continue
                        if inOrder(c2, c6, c5, fwd, tour): yield (c1, c2, c3, c4, c5, c6), TO_126534, G3a 
                        else: yield (c1, c2, c3, c4, c5, c6), TO_125634, G3a 

                if tourOrderPrev == TO_1243:
                    if inOrder(c3, c5, c1, fwd, tour):
                        if inOrder(c3, c5, c6, fwd, tour):
                            if (c5 == c1): continue
                            if (c6 == c2): continue
                            if (c5 == c2): continue 
                            if (c6 == c1): continue
                            yield (c1, c2, c3, c4, c5, c6), TO_124356, G3a  # disconnecting, but starts 4-opt
                        else: yield (c1, c2, c3, c4, c5, c6), TO_124365, G3a 
                    else:
                        if inOrder(c2, c5, c6, fwd, tour):
                            if (c5 == c1): continue
                            if (c6 == c2): continue
                            if (c5 == c2): continue
                            if (c6 == c1): continue
                            yield (c1, c2, c3, c4, c5, c6), TO_125643, G3a 
                        else: yield (c1, c2, c3, c4, c5, c6), TO_126543, G3a  # disconnecting, but starts 4-opt
                
                else: continue # no more possibilities


def LK_4Move(tour,  nearest_neighbors, start, stop): # return 8 cities, tourOrder, and binary flag (True if valid 4 opt move)
    # http://tsp-basics.blogspot.com/2017/06/lin-kernighan-algorithm-basics-part-5.html
  nb_nn_4opt = 5
  for (c1, c2, c3, c4, c5, c6) ,tourOrderPrev, G3a in LK_3Move(tour,  nearest_neighbors, start, stop):
    fwd = (c2 == t_succ(c1, tour))
    c6_succ = t_succ(c6, tour)
    c6_pred = t_pred(c6, tour)

    for c7 in nearest_neighbors[c6][:nb_nn_4opt]:
      if (c7 == c6_succ):continue
      if (c7 == c6_pred):continue
      if (c6 == c2): continue
      if (c7 == c3): continue
      if (c6 == c3): continue
      if (c7 == c2): continue

      G3 = G3a - distance(c6, c7)
      if G3  <= 0: break

      c7_succ = t_succ(c7, tour)
      c7_pred = t_pred(c7, tour)
      for c8 in [c7_succ, c7_pred]:
        if (c7 == c1):continue 
        if (c8 == c2):continue 
        if (c7 == c2):continue 
        if (c8 == c1):continue
        if (c7 == c3):continue 
        if (c8 == c4):continue
        if (c7 == c4):continue
        if (c8 == c3):continue

        G4a = G3 + distance(c7, c8)
        if tourOrderPrev == TO_126534:
          if inOrder(c4, c7, c1, fwd, tour):
             if inOrder(c4, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12653478,  False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12653487, True, G4a
          elif inOrder(c5, c7, c3, fwd, tour):
             if inOrder(c5, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12657834,  True, G4a
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12658734, False, G4a # disconnecting, starts 5-opt
          else: #  inOrder(c2, c7, c6, fwd):
             if inOrder(c2, c7, c8, fwd, tour):yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12786534,  True, G4a
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12876534, False, G4a # disconnecting, starts 5-opt


        if tourOrderPrev == TO_125634:
          if inOrder(c4, c7, c1, fwd, tour):
             if inOrder(c4, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12563478, False, G4a # disconnecting, starts 5-opt
             else:yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12563487, True, G4a
          elif inOrder(c6, c7, c3, fwd, tour):
             if inOrder(c6, c7, c8, fwd, tour):yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12567834, False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12568734, True, G4a
          else: #  inOrder(c2, c7, c5, fwd):
             if inOrder(c2, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12785634, False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12875634, True, G4a

        if tourOrderPrev == TO_123456:
            if inOrder(c6, c7, c1, fwd, tour): continue
            elif inOrder(c4, c7, c5, fwd, tour):
               if inOrder(c4, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12347856, False, G4a # disconnecting, starts 5-opt
               else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12348756, False, G4a # disconnecting, starts 5-opt
            else: #  inOrder(c2, c7, c3, fwd):
               if inOrder(c2, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12783456, False, G4a # disconnecting, starts 5-opt
               else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12873456, False, G4a # disconnecting, starts 5-opt

        if tourOrderPrev == TO_125643:
          if inOrder(c3, c7, c1, fwd, tour):
             if inOrder(c3, c7, c8, fwd, tour):
                  yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12564378, False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12564387, True, G4a
          elif inOrder(c6, c7, c4, fwd, tour):
             if inOrder(c6, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12567843, False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12568743, True, G4a
          else: #  inOrder(c2, c7, c5, fwd):
             if inOrder(c2, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12785643, True, G4a
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12875643, False, G4a # disconnecting, starts 5-opt

        if tourOrderPrev == TO_124365:
          if inOrder(c5, c7, c1, fwd, tour):
             if inOrder(c5, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12436578, False, G4a # disconnecting, starts 5-op
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12436587, True, G4a
          elif inOrder(c3, c7, c6, fwd, tour):
             if inOrder(c3, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12437865, True, G4a
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12438765, False, G4a # disconnecting, starts 5-opt

          else: #  inOrder(c2, c7, c4, fwd):
             if inOrder(c2, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12784365, False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12874365, True, G4a

        if tourOrderPrev == TO_123465:
          if inOrder(c5, c7, c1, fwd, tour):          
             if inOrder(c5, c7, c8, fwd, tour): continue
             else:yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12346587, False, G4a # disconnecting, starts 5-opt
          elif inOrder(c4, c7, c6, fwd, tour):
             if inOrder(c4, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12347865, False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12348765, False, G4a # disconnecting, starts 5-opt

          else: #  inOrder(c2, c7, c3, fwd):
             if inOrder(c2, c7, c8, fwd, tour): yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12783465, True, G4a
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12873465, True, G4a

        if tourOrderPrev == TO_126543:
          if inOrder(c3, c7, c1, fwd, tour):
               if inOrder(c3, c7, c8, fwd, tour):continue
               else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12654387, False, G4a # disconnecting, starts 5-opt

          elif inOrder(c5, c7, c4, fwd, tour):
             if inOrder(c5, c7, c8, fwd, tour):yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12657843, True, G4a
             else:yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12658743, True, G4a
          else: #  inOrder(c2, c7, c6, fwd):
             if inOrder(c2, c7, c8, fwd, tour):yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12786543, False, G4a # disconnecting, starts 5-opt
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12876543, False, G4a # disconnecting, starts 5-opt


        if tourOrderPrev == TO_124356:
          if inOrder(c6, c7, c1, fwd, tour):
             if inOrder(c6, c7, c8, fwd, tour): continue
             else: yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12435687, False, G4a # disconnecting, starts 5-opt
          elif inOrder(c3, c7, c5, fwd, tour):
             if inOrder(c3, c7, c8, fwd, tour):yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12437856, True, G4a
             else:yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12438756, True, G4a
          else: #  inOrder(c2, c7, c4, fwd):
             if inOrder(c2, c7, c8, fwd, tour):yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12784356, True, G4a
             else:yield [c1,c2,c3,c4,c5,c6,c7,c8], TO_12874356, True, G4a
        else: continue


   

def LK_5Move(tour, nearest_neighbors, start, stop):
    # Source: http://tsp-basics.blogspot.com/2017/07/extending-lk-5-opt-move-part-1.html
  nb_nn_5opt = 5
  for [c1, c2, c3, c4, c5, c6, c7, c8], tourOrderPrev, valid_4_opt_flag, G4a in LK_4Move(tour, nearest_neighbors, start, stop):
    fwd = (c2 == t_succ(c1, tour))
    #goodSufficesList = []  # empty list (sequence)
    c8_succ = t_succ(c8, tour)
    c8_pred = t_pred(c8, tour)
    for c9 in nearest_neighbors[c8][:nb_nn_5opt]:
        #if tried_c9 >= 2*Max_Breadth_4:
        #break find_promising_moves
        if (c9 == c8_succ): continue
        if (c9 == c8_pred): continue
        if (c8 == c2): continue
        if (c9 == c3): continue
        if (c8 == c3): continue
        if (c9 == c2): continue
        if (c8 == c4): continue
        if (c9 == c5): continue
        if (c8 == c5): continue
        if (c9 == c4): continue
        if (c9 == c1): continue

        G4 = G4a - distance(c8, c9)
        if G4 < 0: break

        c9_succ = t_succ(c9, tour)
        c9_pred = t_pred(c9, tour)

        for c10 in [c9_succ, c9_pred]:
            if (c10 == c1): continue
            if (c9 == c3): continue
            if (c10 == c4) : continue
            if (c9 == c4): continue
            if (c10 == c3): continue
            if (c9 == c5): continue 
            if (c10 == c6): continue
            if (c9 == c6): continue 
            if (c10 == c5): continue

            if tourOrderPrev == TO_12347856:
                if inOrder(c2, c9, c3, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A347856
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9347856

            if tourOrderPrev == TO_12348756:
                if inOrder(c2, c9, c3, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A348756
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9348756

            if tourOrderPrev == TO_12783456:
                if inOrder(c4, c9, c5, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278349A56
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_127834A956

            if tourOrderPrev == TO_12873456:
                if inOrder(c4, c9, c5, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287349A56
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_128734A956

            if tourOrderPrev == TO_12346587:
                if inOrder(c2, c9, c3, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A346587
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9346587

            if tourOrderPrev == TO_12347865:
                if inOrder(c2, c9, c3, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A347865
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9347865

            if tourOrderPrev == TO_12783465:
                if inOrder(c5, c9, c1, fwd, tour):
                    if inOrder(c5, c10, c9, fwd, tour):
                       yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12783465A9
                elif inOrder(c4, c9, c6, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278349A65
                elif inOrder(c8, c9, c3, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278A93465
                else: #  inOrder(c2, c9, c7, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9783465

            if tourOrderPrev == TO_12873465:
                if inOrder(c5, c9, c1, fwd, tour):
                    if inOrder(c5, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12873465A9
                elif inOrder(c4, c9, c6, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287349A65
                elif inOrder(c7, c9, c3, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12879A3465
                else: #  inOrder(c2, c9, c8, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A873465

            if tourOrderPrev == TO_12563478:
                if inOrder(c4, c9, c7, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256349A78
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_125634A978
                elif inOrder(c6, c9, c3, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A3478
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256A93478
                elif inOrder(c2, c9, c5, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A563478
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9563478

            if tourOrderPrev == TO_12563487:
                if inOrder(c7, c9, c1, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12563487A9
                elif inOrder(c4, c9, c8, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256349A87
                elif inOrder(c6, c9, c3, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A3487
                else: #  inOrder(c2, c9, c5, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A563487

            if tourOrderPrev == TO_12785634:
                if inOrder(c6, c9, c3, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278569A34
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_127856A934
                elif inOrder(c2, c9, c7, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A785634
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9785634

            if tourOrderPrev == TO_12875634:
                if inOrder(c4, c9, c1, fwd, tour):
                    if inOrder(c4, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12875634A9
                elif inOrder(c6, c9, c3, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287569A34
                elif inOrder(c7, c9, c5, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287A95634
                else: #  inOrder(c2, c9, c8, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A875634

            if tourOrderPrev == TO_12567834:
                if inOrder(c6, c9, c7, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A7834
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256A97834

            if tourOrderPrev == TO_12568734:
                if inOrder(c4, c9, c1, fwd, tour):
                    if inOrder(c4, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12568734A9
                elif inOrder(c7, c9, c3, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_125687A934
                elif inOrder(c6, c9, c8, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A8734
                else: #  inOrder(c2, c9, c5, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9568734

            if tourOrderPrev == TO_12653478:
                if inOrder(c4, c9, c7, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265349A78
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_126534A978
                elif inOrder(c5, c9, c3, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12659A3478
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265A93478
                elif inOrder(c2, c9, c6, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A653478
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9653478

            if tourOrderPrev == TO_12653487:
                if inOrder(c7, c9, c1, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12653487A9
                elif inOrder(c4, c9, c8, fwd, tour):
                    if inOrder(c4, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265349A87
                elif inOrder(c5, c9, c3, fwd, tour):
                    if inOrder(c5, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265A93487
                else: #  inOrder(c2, c9, c6, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9653487

            if tourOrderPrev == TO_12657834:
                if inOrder(c4, c9, c1, fwd, tour):
                    if inOrder(c4, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12657834A9
                elif inOrder(c8, c9, c3, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_126578A934
                elif inOrder(c5, c9, c7, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12659A7834
                else: #  inOrder(c2, c9, c6, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9657834

            if tourOrderPrev == TO_12658734:
                if inOrder(c7, c9, c3, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265879A34
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_126587A934
                elif inOrder(c2, c9, c6, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A658734
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9658734

            if tourOrderPrev == TO_12786534:
                if inOrder(c4, c9, c1, fwd, tour):
                    if inOrder(c4, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12786534A9
                elif inOrder(c5, c9, c3, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278659A34
                elif inOrder(c8, c9, c6, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278A96534
                else: #  inOrder(c2, c9, c7, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A786534

            if tourOrderPrev == TO_12876534:
                if inOrder(c7, c9, c6, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12879A6534
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287A96534

            if tourOrderPrev == TO_12435687:
                if inOrder(c3, c9, c5, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12439A5687
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243A95687
                elif inOrder(c2, c9, c4, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A435687
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9435687

            if tourOrderPrev == TO_12437856:
                if inOrder(c6, c9, c1, fwd, tour):
                    if inOrder(c6, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12437856A9
                elif inOrder(c8, c9, c5, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_124378A956
                elif inOrder(c3, c9, c7, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243A97856
                else: #  inOrder(c2, c9, c4, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A437856

            if tourOrderPrev == TO_12438756:
                if inOrder(c6, c9, c1, fwd, tour):
                    if inOrder(c6, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12438756A9
                elif inOrder(c7, c9, c5, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243879A56
                elif inOrder(c3, c9, c8, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12439A8756
                else: #  inOrder(c2, c9, c4, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9438756

            if tourOrderPrev == TO_12784356:
                if inOrder(c6, c9, c1, fwd, tour):
                    if inOrder(c6, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12784356A9
                elif inOrder(c3, c9, c5, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278439A56
                elif inOrder(c8, c9, c4, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278A94356
                else: #  inOrder(c2, c9, c7, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9784356

            if tourOrderPrev == TO_12874356:
                if inOrder(c6, c9, c1, fwd, tour):
                    if inOrder(c6, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12874356A9
                elif inOrder(c3, c9, c5, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_128743A956
                elif inOrder(c7, c9, c4, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12879A4356
                else: #  inOrder(c2, c9, c8, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A874356

            if tourOrderPrev == TO_12436578:
                if inOrder(c5, c9, c7, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243659A78
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_124365A978
                elif inOrder(c3, c9, c6, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12439A6578
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243A96578
                elif inOrder(c2, c9, c4, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A436578
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9436578

            if tourOrderPrev == TO_12436587:
                if inOrder(c7, c9, c1, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12436587A9
                elif inOrder(c5, c9, c8, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243659A87
                elif inOrder(c3, c9, c6, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243A96587
                else: #  inOrder(c2, c9, c4, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A436587

            if tourOrderPrev == TO_12437865:
                if inOrder(c5, c9, c1, fwd, tour):
                    if inOrder(c5, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12437865A9
                elif inOrder(c8, c9, c6, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_124378A965
                elif inOrder(c3, c9, c7, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12439A7865
                else: #  inOrder(c2, c9, c4, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9437865

            if tourOrderPrev == TO_12438765:
                if inOrder(c7, c9, c6, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1243879A65
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_124387A965

            if tourOrderPrev == TO_12784365:
                if inOrder(c3, c9, c6, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278439A65
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_127843A965
                elif inOrder(c2, c9, c7, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A784365
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9784365

            if tourOrderPrev == TO_12874365:
                if inOrder(c5, c9, c1, fwd, tour):
                    if inOrder(c5, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12874365A9
                elif inOrder(c3, c9, c6, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_128743A965
                elif inOrder(c7, c9, c4, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287A94365
                else: #  inOrder(c2, c9, c8, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A874365

            if tourOrderPrev == TO_12564378:
                if inOrder(c3, c9, c7, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256439A78
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_125643A978
                elif inOrder(c6, c9, c4, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A4378
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256A94378
                elif inOrder(c2, c9, c5, fwd, tour):
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A564378
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9564378

            if tourOrderPrev == TO_12564387:
                if inOrder(c7, c9, c1, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12564387A9
                elif inOrder(c3, c9, c8, fwd, tour):
                    if inOrder(c3, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256439A87
                elif inOrder(c6, c9, c4, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A4387
                else: #  inOrder(c2, c9, c5, fwd, tour)
                    if inOrder(c2, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12A9564387

            if tourOrderPrev == TO_12567843:
                if inOrder(c6, c9, c7, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A7843
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1256A97843

            if tourOrderPrev == TO_12568743:
                if inOrder(c3, c9, c1, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12568743A9
                elif inOrder(c7, c9, c4, fwd, tour):
                    if inOrder(c7, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_125687A943
                elif inOrder(c6, c9, c8, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12569A8743
                else: #  inOrder(c2, c9, c5, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A568743

            if tourOrderPrev == TO_12785643:
                if inOrder(c3, c9, c1, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12785643A9
                elif inOrder(c6, c9, c4, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278569A43
                elif inOrder(c8, c9, c5, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278A95643
                else: #  inOrder(c2, c9, c7, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A785643

            if tourOrderPrev == TO_12875643:
                if inOrder(c6, c9, c4, fwd, tour):
                    if inOrder(c6, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287569A43
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_128756A943
                elif inOrder(c7, c9, c5, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12879A5643
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1287A95643

            if tourOrderPrev == TO_12654387:
                if inOrder(c5, c9, c4, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12659A4387
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265A94387

            if tourOrderPrev == TO_12657843:
                if inOrder(c3, c9, c1, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12657843A9
                elif inOrder(c8, c9, c4, fwd, tour):
                    if inOrder(c8, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_126578A943
                elif inOrder(c5, c9, c7, fwd, tour):
                    if inOrder(c5, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265A97843
                else: #  inOrder(c2, c9, c6, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A657843

            if tourOrderPrev == TO_12658743:
                if inOrder(c3, c9, c1, fwd, tour):
                    if inOrder(c3, c10, c9, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12658743A9
                elif inOrder(c7, c9, c4, fwd, tour):
                    if inOrder(c7, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1265879A43
                elif inOrder(c5, c9, c8, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_12659A8743
                else: #  inOrder(c2, c9, c6, fwd, tour)
                    if inOrder(c2, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_129A658743

            if tourOrderPrev == TO_12786543:
                if inOrder(c5, c9, c4, fwd, tour):
                    if inOrder(c5, c9, c10, fwd, tour):
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_1278659A43
                    else:
                        yield [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], TO_127865A943
            else: continue # inspect next c10

N = 197769

TO_00 = 0  # unknown or unmaintained sequence
TO_1234 = 1
TO_1243 = 2  # 2-opt
# descendants of TO_1234
TO_123456 = 3
TO_123465 = 4
TO_125634 = 5  # 3-opt
TO_126534 = 6  # 3-opt
# descendants of TO_1243
TO_124356 = 7
TO_124365 = 8  # 3-opt
TO_125643 = 9  # 3-opt
TO_126543 = 10

TO_12874356 = 'TO_12874356'
# descendants of TO_123456
TO_12345678 = 'TO_12345678'
TO_12345687 = 'TO_12345687'
TO_12347856 = 'TO_12347856'
TO_12348756 = 'TO_12348756'
TO_12783456 = 'TO_12783456'
TO_12873456 = 'TO_12873456'
# descendants of TO_123465
TO_12346578 = 'TO_12346578'
TO_12346587 = 'TO_12346587'
TO_12347865 = 'TO_12347865'
TO_12348765 = 'TO_12348765'
TO_12783465 = 'TO_12783465'  # 4-opt 1
TO_12873465 = 'TO_12873465'  # 4-opt 2
# descendants of TO_125634
TO_12563478 = 'TO_12563478'
TO_12563487 = 'TO_12563487'  # 4-opt 3
TO_12785634 = 'TO_12785634'
TO_12875634 = 'TO_12875634'  # 4-opt 4
TO_12567834 = 'TO_12567834'
TO_12568734 = 'TO_12568734'  # 4-opt 5
# descendants of TO_126534
TO_12653478 = 'TO_12653478'
TO_12653487 = 'TO_12653487'  # 4-opt 6
TO_12657834 = 'TO_12657834'  # 4-opt 7
TO_12658734 = 'TO_12658734'
TO_12786534 = 'TO_12786534'  # 4-opt 8
TO_12876534 = 'TO_12876534'
# descendants of TO_124356
TO_12435678 = 'TO_12435678'
TO_12435687 = 'TO_12435687'
TO_12437856 = 'TO_12437856'  # 4-opt 9
TO_12438756 = 'TO_12438756'  # 4-opt 10
TO_12784356 = 'TO_12784356'  # 4-opt 11
TO_12874356 = 'TO_12874356'  # 4-opt 12
# descendants of TO_124365
TO_12436578 = 'TO_12436578'
TO_12436587 = 'TO_12436587'  # 4-opt 13 
TO_12437865 = 'TO_12437865'  # 4-opt 14
TO_12438765 = 'TO_12438765'
TO_12784365 = 'TO_12784365'
TO_12874365 = 'TO_12874365'  # 4-opt 15
# descendants of TO_125643
TO_12564378 = 'TO_12564378'
TO_12564387 = 'TO_12564387'  # 4-opt 16 
TO_12567843 = 'TO_12567843'
TO_12568743 = 'TO_12568743'  # 4-opt 17
TO_12785643 = 'TO_12785643'  # 4-opt 18
TO_12875643 = 'TO_12875643'
# descendants of TO_126543
TO_12654378 = 'TO_12654378'
TO_12654387 = 'TO_12654387'
TO_12657843 = 'TO_12657843'  # 4-opt 19
TO_12658743 = 'TO_12658743'  # 4-opt 20
TO_12786543 = 'TO_12786543'
TO_12876543 = 'TO_12876543'

VALID_4_OPT = [TO_12783465,TO_12873465,TO_12563487,TO_12875634,TO_12568734,TO_12653487,TO_12657834,TO_12786534,
TO_12437856,TO_12438756,TO_12784356,TO_12874356,TO_12436587,TO_12437865,TO_12874365,TO_12564387,TO_12568743,TO_12785643,TO_12657843,TO_12658743]


# descendants of TO_12345678
TO_123456789A = 'TO_123456789A'
TO_12345678A9 = 'TO_12345678A9'
TO_1234569A78 = 'TO_1234569A78'
TO_123456A978 = 'TO_123456A978'
TO_12349A5678 = 'TO_12349A5678'
TO_1234A95678 = 'TO_1234A95678'
TO_129A345678 = 'TO_129A345678'
TO_12A9345678 = 'TO_12A9345678'
# descendants of TO_12345687
TO_123456879A = 'TO_123456879A'
TO_12345687A9 = 'TO_12345687A9'
TO_1234569A87 = 'TO_1234569A87'
TO_123456A987 = 'TO_123456A987'
TO_12349A5687 = 'TO_12349A5687'
TO_1234A95687 = 'TO_1234A95687'
TO_129A345687 = 'TO_129A345687'
TO_12A9345687 = 'TO_12A9345687'
# descendants of TO_12347856
TO_123478569A = 'TO_123478569A'
TO_12347856A9 = 'TO_12347856A9'
TO_1234789A56 = 'TO_1234789A56'
TO_123478A956 = 'TO_123478A956'
TO_12349A7856 = 'TO_12349A7856'
TO_1234A97856 = 'TO_1234A97856'
TO_129A347856 = 'TO_129A347856'  # 5-opt
TO_12A9347856 = 'TO_12A9347856'  # 5-opt
# descendants of TO_12348756
TO_123487569A = 'TO_123487569A'
TO_12348756A9 = 'TO_12348756A9'
TO_1234879A56 = 'TO_1234879A56'
TO_123487A956 = 'TO_123487A956'
TO_12349A8756 = 'TO_12349A8756'
TO_1234A98756 = 'TO_1234A98756'
TO_129A348756 = 'TO_129A348756'  # 5-opt
TO_12A9348756 = 'TO_12A9348756'  # 5-opt
# descendants of TO_12783456
TO_127834569A = 'TO_127834569A'
TO_12783456A9 = 'TO_12783456A9'
TO_1278349A56 = 'TO_1278349A56'  # 5-opt
TO_127834A956 =   'TO_127834A956'# 5-opt
TO_12789A3456 = 'TO_12789A3456'
TO_1278A93456 = 'TO_1278A93456'
TO_129A783456 = 'TO_129A783456'
TO_12A9783456 = 'TO_12A9783456'
# descendants of TO_12873456
TO_128734569A = 'TO_128734569A'
TO_12873456A9 = 'TO_12873456A9'
TO_1287349A56 =   'TO_1287349A56'# 5-opt
TO_128734A956 = 'TO_128734A956'  # 5-opt
TO_12879A3456 = 'TO_12879A3456'
TO_1287A93456 = 'TO_1287A93456'
TO_129A873456 = 'TO_129A873456'
TO_12A9873456 = 'TO_12A9873456'
  # descendants of TO_12346578
TO_123465789A = 'TO_123465789A'
TO_12346578A9 = 'TO_12346578A9'
TO_1234659A78 = 'TO_1234659A78'
TO_123465A978 = 'TO_123465A978'
TO_12349A6578 = 'TO_12349A6578'
TO_1234A96578 = 'TO_1234A96578'
TO_129A346578 = 'TO_129A346578'
TO_12A9346578 = 'TO_12A9346578'
  # descendants of TO_12346587
TO_123465879A = 'TO_123465879A'
TO_12346587A9 = 'TO_12346587A9'
TO_1234659A87 = 'TO_1234659A87'
TO_123465A987 = 'TO_123465A987'
TO_12349A6587 = 'TO_12349A6587'
TO_1234A96587 = 'TO_1234A96587'
TO_129A346587 = 'TO_129A346587'  # 5-opt
TO_12A9346587 =  'TO_12A9346587' # 5-opt
  # descendants of TO_12347865
TO_123478659A = 'TO_123478659A'
TO_12347865A9 = 'TO_12347865A9'
TO_1234789A65 = 'TO_1234789A65'
TO_123478A965 = 'TO_123478A965'
TO_12349A7865 = 'TO_12349A7865'
TO_1234A97865 = 'TO_1234A97865'
TO_129A347865 = 'TO_129A347865'  # 5-opt
TO_12A9347865 = 'TO_12A9347865'  # 5-opt
# descendants of TO_12348765
TO_123487659A = 'TO_123487659A'
TO_12348765A9 = 'TO_12348765A9'
TO_1234879A65 = 'TO_1234879A65'
TO_123487A965 = 'TO_123487A965'
TO_12349A8765 = 'TO_12349A8765'
TO_1234A98765 = 'TO_1234A98765'
TO_129A348765 = 'TO_129A348765'
TO_12A9348765 = 'TO_12A9348765'
 # descendants of TO_12783465
TO_127834659A = 'TO_127834659A'
TO_12783465A9 = 'TO_12783465A9'  # 5-opt
TO_1278349A65 = 'TO_1278349A65'  # 5-opt
TO_127834A965 = 'TO_127834A965'
TO_12789A3465 = 'TO_12789A3465'
TO_1278A93465 = 'TO_1278A93465'  # 5-opt
TO_129A783465 = 'TO_129A783465' 
TO_12A9783465 = 'TO_12A9783465'  # 5-opt
# descendants of TO_12873465
TO_128734659A = 'TO_128734659A'
TO_12873465A9 = 'TO_12873465A9'  # 5-opt
TO_1287349A65 =  'TO_1287349A65' # 5-opt
TO_128734A965 = 'TO_128734A965'
TO_12879A3465 = 'TO_12879A3465'  # 5-opt
TO_1287A93465 = 'TO_1287A93465'
TO_129A873465 =  'TO_129A873465' # 5-opt
TO_12A9873465 = 'TO_12A9873465'
# descendants of TO_12563478
TO_125634789A = 'TO_125634789A'
TO_12563478A9 = 'TO_12563478A9'
TO_1256349A78 = 'TO_1256349A78'  # 5-opt
TO_125634A978 = 'TO_125634A978'  # 5-opt
TO_12569A3478 = 'TO_12569A3478'  # 5-opt
TO_1256A93478 = 'TO_1256A93478'  # 5-opt
TO_129A563478 = 'TO_129A563478'  # 5-opt
TO_12A9563478 = 'TO_12A9563478'  # 5-opt
# descendants of TO_12563487
TO_125634879A = 'TO_125634879A'
TO_12563487A9 = 'TO_12563487A9'  # 5-opt
TO_1256349A87 = 'TO_1256349A87'  # 5-opt
TO_125634A987 = 'TO_125634A987'
TO_12569A3487 = 'TO_12569A3487'  # 5-opt
TO_1256A93487 = 'TO_1256A93487' 
TO_129A563487 =  'TO_129A563487' # 5-opt
TO_12A9563487 = 'TO_12A9563487'
# descendants of TO_12785634
TO_127856349A = 'TO_127856349A'
TO_12785634A9 = 'TO_12785634A9'
TO_1278569A34 = 'TO_1278569A34'  # 5-opt
TO_127856A934 = 'TO_127856A934'  # 5-opt
TO_12789A5634 = 'TO_12789A5634'
TO_1278A95634 = 'TO_1278A95634'
TO_129A785634 = 'TO_129A785634'  # 5-opt
TO_12A9785634 = 'TO_12A9785634'  # 5-opt
# descendants of TO_12875634
TO_128756349A = 'TO_128756349A'
TO_12875634A9 =  'TO_12875634A9' # 5-opt
TO_1287569A34 =  'TO_1287569A34' # 5-opt
TO_128756A934 = 'TO_128756A934'
TO_12879A5634 = 'TO_12879A5634'
TO_1287A95634 = 'TO_1287A95634'  # 5-opt
TO_129A875634 =   'TO_129A875634'# 5-opt
TO_12A9875634 = 'TO_12A9875634'
# descendants of TO_12567834
TO_125678349A = 'TO_125678349A'
TO_12567834A9 = 'TO_12567834A9'
TO_1256789A34 = 'TO_1256789A34'
TO_125678A934 = 'TO_125678A934'
TO_12569A7834 =  'TO_12569A7834' # 5-opt
TO_1256A97834 = 'TO_1256A97834'  # 5-opt
TO_129A567834 = 'TO_129A567834'
TO_12A9567834 = 'TO_12A9567834'
# descendants of TO_12568734
TO_125687349A = 'TO_125687349A'
TO_12568734A9 = 'TO_12568734A9'  # 5-opt
TO_1256879A34 = 'TO_1256879A34'
TO_125687A934 = 'TO_125687A934'  # 5-opt
TO_12569A8734 = 'TO_12569A8734'  # 5-opt
TO_1256A98734 = 'TO_1256A98734'
TO_129A568734 = 'TO_129A568734'
TO_12A9568734 = 'TO_12A9568734'  # 5-opt
# descendants of TO_12653478
TO_126534789A = 'TO_126534789A'
TO_12653478A9 = 'TO_12653478A9'
TO_1265349A78 = 'TO_1265349A78'  # 5-opt
TO_126534A978 = 'TO_126534A978'  # 5-opt
TO_12659A3478 =  'TO_12659A3478' # 5-opt
TO_1265A93478 = 'TO_1265A93478'  # 5-opt
TO_129A653478 = 'TO_129A653478'  # 5-opt
TO_12A9653478 = 'TO_12A9653478'  # 5-opt
# descendants of TO_12653487
TO_126534879A = 'TO_126534879A'
TO_12653487A9 = 'TO_12653487A9'  # 5-opt
TO_1265349A87 = 'TO_1265349A87'  # 5-opt
TO_126534A987 = 'TO_126534A987'
TO_12659A3487 = 'TO_12659A3487'
TO_1265A93487 =  'TO_1265A93487' # 5-opt
TO_129A653487 = 'TO_129A653487'
TO_12A9653487 = 'TO_12A9653487'  # 5-opt
# descendants of TO_12657834
TO_126578349A = 'TO_126578349A'
TO_12657834A9 = 'TO_12657834A9'  # 5-opt
TO_1265789A34 = 'TO_1265789A34'
TO_126578A934 =  'TO_126578A934' # 5-opt
TO_12659A7834 =   'TO_12659A7834'# 5-opt
TO_1265A97834 = 'TO_1265A97834'
TO_129A657834 = 'TO_129A657834'
TO_12A9657834 = 'TO_12A9657834'  # 5-opt
# descendants of TO_12658734
TO_126587349A = 'TO_126587349A'
TO_12658734A9 = 'TO_12658734A9'
TO_1265879A34 = 'TO_1265879A34'  # 5-opt
TO_126587A934 =  'TO_126587A934' # 5-opt
TO_12659A8734 = 'TO_12659A8734'
TO_1265A98734 = 'TO_1265A98734'
TO_129A658734 = 'TO_129A658734'  # 5-opt
TO_12A9658734 = 'TO_12A9658734'  # 5-opt
# descendants of TO_12786534
TO_127865349A = 'TO_127865349A'
TO_12786534A9 = 'TO_12786534A9'  # 5-opt
TO_1278659A34 = 'TO_1278659A34'  # 5-opt
TO_127865A934 = 'TO_127865A934'
TO_12789A6534 = 'TO_12789A6534'
TO_1278A96534 = 'TO_1278A96534'  # 5-opt
TO_129A786534 = 'TO_129A786534'  # 5-opt
TO_12A9786534 = 'TO_12A9786534'
# descendants of TO_12876534
TO_128765349A = 'TO_128765349A'
TO_12876534A9 = 'TO_12876534A9'
TO_1287659A34 = 'TO_1287659A34'
TO_128765A934 = 'TO_128765A934'
TO_12879A6534 = 'TO_12879A6534'  # 5-opt
TO_1287A96534 = 'TO_1287A96534'  # 5-opt
TO_129A876534 = 'TO_129A876534'
TO_12A9876534 = 'TO_12A9876534'
# descendants of TO_12435678
TO_124356789A = 'TO_124356789A'
TO_12435678A9 = 'TO_12435678A9'
TO_1243569A78 = 'TO_1243569A78'
TO_124356A978 = 'TO_124356A978'
TO_12439A5678 = 'TO_12439A5678'
TO_1243A95678 = 'TO_1243A95678'
TO_129A435678 = 'TO_129A435678'
TO_12A9435678 = 'TO_12A9435678'
# descendants of TO_12435687
TO_124356879A = 'TO_124356879A'
TO_12435687A9 = 'TO_12435687A9'
TO_1243569A87 = 'TO_1243569A87'
TO_124356A987 = 'TO_124356A987'
TO_12439A5687 = 'TO_12439A5687'  # 5-opt
TO_1243A95687 = 'TO_1243A95687'  # 5-opt
TO_129A435687 = 'TO_129A435687'  # 5-opt
TO_12A9435687 = 'TO_12A9435687'  # 5-opt
# descendants of TO_12437856
TO_124378569A = 'TO_124378569A'
TO_12437856A9 = 'TO_12437856A9'  # 5-opt
TO_1243789A56 = 'TO_1243789A56'
TO_124378A956 = 'TO_124378A956'  # 5-opt
TO_12439A7856 = 'TO_12439A7856'
TO_1243A97856 = 'TO_1243A97856'  # 5-opt
TO_129A437856 = 'TO_129A437856'  # 5-opt
TO_12A9437856 = 'TO_12A9437856'
# descendants of TO_12438756
TO_124387569A = 'TO_124387569A'
TO_12438756A9 = 'TO_12438756A9'  # 5-opt
TO_1243879A56 = 'TO_1243879A56'  # 5-opt
TO_124387A956 = 'TO_1243879A56'
TO_12439A8756 = 'TO_1243879A56'  # 5-opt
TO_1243A98756 = 'TO_1243A98756'
TO_129A438756 = 'TO_129A438756'
TO_12A9438756 = 'TO_12A9438756'  # 5-opt
# descendants of TO_12784356
TO_127843569A = 'TO_127843569A'
TO_12784356A9 = 'TO_12784356A9'  # 5-opt
TO_1278439A56 =  'TO_1278439A56' # 5-opt
TO_127843A956 = 'TO_127843A956'
TO_12789A4356 = 'TO_12789A4356'
TO_1278A94356 =  'TO_1278A94356' # 5-opt
TO_129A784356 ='TO_129A784356' 
TO_12A9784356 = 'TO_12A9784356'  # 5-opt
# descendants of TO_12874356
TO_128743569A = 'TO_128743569A'
TO_12874356A9 =  'TO_12874356A9' # 5-opt
TO_1287439A56 ='TO_1287439A56' 
TO_128743A956 = 'TO_128743A956'  # 5-opt
TO_12879A4356 = 'TO_12879A4356'  # 5-opt
TO_1287A94356 = 'TO_1287A94356'
TO_129A874356 =  'TO_129A874356' # 5-opt
TO_12A9874356 = 'TO_12A9874356'
# descendants of TO_12436578
TO_124365789A = 'TO_124365789A'
TO_12436578A9 = 'TO_12436578A9'
TO_1243659A78 = 'TO_1243659A78'  # 5-opt
TO_124365A978 = 'TO_124365A978'  # 5-opt
TO_12439A6578 = 'TO_12439A6578'  # 5-opt
TO_1243A96578 =  'TO_1243A96578' # 5-opt
TO_129A436578 =  'TO_129A436578' # 5-opt
TO_12A9436578 = 'TO_12A9436578'  # 5-opt
# descendants of TO_12436587
TO_124365879A = 'TO_124365879A'
TO_12436587A9 = 'TO_12436587A9'  # 5-opt
TO_1243659A87 =  'TO_1243659A87' # 5-opt
TO_124365A987 = 'TO_124365A987'
TO_12439A6587 = 'TO_12439A6587'
TO_1243A96587 = 'TO_1243A96587'  # 5-opt
TO_129A436587 =  'TO_129A436587' # 5-opt
TO_12A9436587 = 'TO_12A9436587'
# descendants of TO_12437865
TO_124378659A = 'TO_124378659A'
TO_12437865A9 = 'TO_12437865A9'  # 5-opt
TO_1243789A65 = 'TO_1243789A65'
TO_124378A965 = 'TO_124378A965'  # 5-opt
TO_12439A7865 = 'TO_12439A7865'  # 5-opt
TO_1243A97865 = 'TO_1243A97865'
TO_129A437865 = 'TO_129A437865'
TO_12A9437865 = 'TO_12A9437865'  # 5-opt
# descendants of TO_12438765
TO_124387659A = 'TO_124387659A'
TO_12438765A9 = 'TO_12438765A9'
TO_1243879A65 = 'TO_1243879A65'  # 5-opt
TO_124387A965 = 'TO_124387A965'  # 5-opt
TO_12439A8765 = 'TO_12439A8765'
TO_1243A98765 = 'TO_1243A98765'
TO_129A438765 = 'TO_129A438765'
TO_12A9438765 = 'TO_12A9438765'
# descendants of TO_12784365
TO_127843659A = 'TO_127843659A'
TO_12784365A9 = 'TO_12784365A9'
TO_1278439A65 = 'TO_1278439A65'  # 5-opt
TO_127843A965 = 'TO_127843A965'  # 5-opt
TO_12789A4365 = 'TO_12789A4365'
TO_1278A94365 = 'TO_1278A94365'
TO_129A784365 = 'TO_129A784365'  # 5-opt
TO_12A9784365 = 'TO_12A9784365'  # 5-opt
# descendants of TO_12874365
TO_128743659A = 'TO_128743659A'
TO_12874365A9 = 'TO_12874365A9'  # 5-opt
TO_1287439A65 = 'TO_1287439A65'
TO_128743A965 = 'TO_128743A965'  # 5-opt
TO_12879A4365 = 'TO_12879A4365'
TO_1287A94365 = 'TO_1287A94365'  # 5-opt
TO_129A874365 = 'TO_129A874365'  # 5-opt
TO_12A9874365 = 'TO_12A9874365'
# descendants of TO_12564378
TO_125643789A = 'TO_125643789A'
TO_12564378A9 = 'TO_12564378A9'
TO_1256439A78 = 'TO_1256439A78'  # 5-opt
TO_125643A978 = 'TO_125643A978'  # 5-opt
TO_12569A4378 = 'TO_12569A4378'  # 5-opt
TO_1256A94378 = 'TO_1256A94378'  # 5-opt
TO_129A564378 = 'TO_129A564378'  # 5-opt
TO_12A9564378 = 'TO_12A9564378'  # 5-opt
# descendants of TO_12564387
TO_125643879A = 'TO_125643879A'
TO_12564387A9 = 'TO_12564387A9'  # 5-opt
TO_1256439A87 = 'TO_1256439A87'  # 5-opt
TO_125643A987 = 'TO_125643A987'
TO_12569A4387 = 'TO_12569A4387'  # 5-opt
TO_1256A94387 = 'TO_1256A94387'
TO_129A564387 = 'TO_129A564387'
TO_12A9564387 =  'TO_12A9564387' # 5-opt
# descendants of TO_12567843
TO_125678439A = 'TO_125678439A'
TO_12567843A9 = 'TO_12567843A9'
TO_1256789A43 = 'TO_1256789A43'
TO_125678A943 = 'TO_125678A943'
TO_12569A7843 = 'TO_12569A7843'  # 5-opt
TO_1256A97843 = 'TO_1256A97843'  # 5-opt
TO_129A567843 = 'TO_129A567843'
TO_12A9567843 = 'TO_12A9567843'
# descendants of TO_12568743
TO_125687439A = 'TO_125687439A'
TO_12568743A9 = 'TO_12568743A9'  # 5-opt
TO_1256879A43 = 'TO_1256879A43'
TO_125687A943 = 'TO_125687A943'  # 5-opt
TO_12569A8743 =  'TO_12569A8743' # 5-opt
TO_1256A98743 = 'TO_1256A98743'
TO_129A568743 = 'TO_129A568743'  # 5-opt
TO_12A9568743 = 'TO_12A9568743'
# descendants of TO_12785643
TO_127856439A = 'TO_127856439A'
TO_12785643A9 = 'TO_12785643A9'  # 5-opt
TO_1278569A43 = 'TO_1278569A43'  # 5-opt
TO_127856A943 = 'TO_127856A943'
TO_12789A5643 = 'TO_12789A5643'
TO_1278A95643 = 'TO_1278A95643'  # 5-opt
TO_129A785643 = 'TO_129A785643'  # 5-opt
TO_12A9785643 = 'TO_12A9785643'
# descendants of TO_12875643
TO_128756439A = 'TO_128756439A'
TO_12875643A9 = 'TO_12875643A9'
TO_1287569A43 = 'TO_1287569A43'  # 5-opt
TO_128756A943 = 'TO_128756A943'  # 5-opt
TO_12879A5643 = 'TO_12879A5643'  # 5-opt
TO_1287A95643 = 'TO_1287A95643'  # 5-opt
TO_129A875643 = 'TO_129A875643'
TO_12A9875643 = 'TO_12A9875643'
# descendants of TO_12654378
TO_126543789A = 'TO_126543789A'
TO_12654378A9 = 'TO_12654378A9'
TO_1265439A78 = 'TO_1265439A78'
TO_126543A978 = 'TO_126543A978'
TO_12659A4378 = 'TO_12659A4378'
TO_1265A94378 = 'TO_1265A94378'
TO_129A654378 = 'TO_129A654378'
TO_12A9654378 = 'TO_12A9654378'
# descendants of TO_12654387
TO_126543879A = 'TO_126543879A'
TO_12654387A9 = 'TO_12654387A9'
TO_1265439A87 = 'TO_1265439A87'
TO_126543A987 = 'TO_126543A987'
TO_12659A4387 =   'TO_12659A4387'# 5-opt
TO_1265A94387 =   'TO_1265A94387'# 5-opt
TO_129A654387 = 'TO_129A654387'
TO_12A9654387 = 'TO_12A9654387'
# descendants of TO_12657843
TO_126578439A = 'TO_126578439A'
TO_12657843A9 =   'TO_12657843A9'# 5-opt
TO_1265789A43 = 'TO_1265789A43'
TO_126578A943 =   'TO_126578A943'# 5-opt
TO_12659A7843 = 'TO_12659A7843'
TO_1265A97843 =   'TO_1265A97843'# 5-opt
TO_129A657843 =   'TO_129A657843'# 5-opt
TO_12A9657843 = 'TO_12A9657843'
# descendants of TO_12658743
TO_126587439A = 'TO_126587439A'
TO_12658743A9 = 'TO_12658743A9'  # 5-opt
TO_1265879A43 =   'TO_1265879A43'# 5-opt
TO_126587A943 = 'TO_126587A943'
TO_12659A8743 =   'TO_12659A8743'# 5-opt
TO_1265A98743 = 'TO_1265A98743'
TO_129A658743 =   'TO_129A658743'# 5-opt
TO_12A9658743 = 'TO_12A9658743'
# descendants of TO_12786543
TO_127865439A = 'TO_127865439A'
TO_12786543A9 = 'TO_12786543A9'
TO_1278659A43 = 'TO_1278659A43'  # 5-opt
TO_127865A943 = 'TO_127865A943'  # 5-opt
TO_12789A6543 = 'TO_12789A6543'
TO_1278A96543 = 'TO_1278A96543'
TO_129A786543 = 'TO_129A786543'
TO_12A9786543 = 'TO_12A9786543'
# descendants of TO_12876543
TO_128765439A = 'TO_128765439A'
TO_12876543A9 = 'TO_12876543A9'
TO_1287659A43 = 'TO_1287659A43'
TO_128765A943 = 'TO_128765A943'
TO_12879A6543 = 'TO_12879A6543'
TO_1287A96543 = 'TO_1287A96543'
TO_129A876543 = 'TO_129A876543'
TO_12A9876543 = 'TO_12A9876543'