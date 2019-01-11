import os
import random
import tsp
import kopt
import datetime

world = tsp.World("resources/cities.csv")
popmusic = tsp.Neighbors.load_pickle("resources/popmusic.pickle")

def optimize_from_scratch(tour, world, neighbors, penalty):
    moves = (
        kopt.Move.move_4opt(neighbors),
        kopt.Move.move_3opt(neighbors),
        )
    tour = kopt.Optimizer.optimize_moves(
        tour, world, moves, percent=penalty)
    return tour

INPUT_TOUR = "resources/tour-raw-1502605.csv"

tour = tsp.Tour.load_csv(INPUT_TOUR)
score = world.score(tour)
print("Input score:", score)

for penalty in [0.01, 0.05, 0.1]: # insert more steps for better final tour
    print("Penalty:", penalty)
    tour = optimize_from_scratch(tour, world, popmusic, penalty)

new_score = world.score(tour)
print('New_score:', new_score)
tsp.Tour.save_csv(tour, "tours/tour-%.4f.csv" % new_score)