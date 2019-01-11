import os
import random
import tsp
import kopt
import datetime

world = tsp.World("resources/cities.csv")
popmusic = tsp.Neighbors.load_pickle("resources/popmusic.pickle")

# Loads the best tour from a directory.
# Relies on the naming convention "tour-<score>.csv".
def best_tour(directory):
    filename = sorted(os.listdir(directory))[0]
    if filename[0] == '.':
        filename = sorted(os.listdir(directory))[1]
    return tsp.Tour.load_csv(os.path.join(directory, filename))

def kick(tour, world, neighbors):
    start = random.randint(0, len(tour) - kick_stride)
    stop = start + kick_stride
    moves = (
        kopt.Move.move_4opt(neighbors),
        kopt.Move.move_3opt(neighbors),
        )

    # Kick
    print("Kicking...")
    for _ in range(kick_rounds):
        move = random.choice(moves)
        new_tour = kopt.Optimizer.optimize_one(
            tour, world, move, min_gain=-kick_loss, start=start, stop=stop)
        if new_tour is not None:
            tour = new_tour

    # Re-optimize
    print("Fixing...")
    reoptimize_moves = moves
    tour = kopt.Optimizer.optimize_moves(
        tour, world, reoptimize_moves, start=start, stop=stop)
    return tour, start, stop

 # The below routine works forever. After each batch if a better tour is found, it will be put to folder "tours"
 # The next batch will load the best tour from that folder.
 # This routine must be stopped by hand.
batch = 0
while True: 
# Kick parameters.
    kick_stride = 2000
    kick_rounds = 40
    kick_loss = random.uniform(0.3,1)

    batch += 1
    print("batch:", batch, ", kick_stride:", kick_stride, ", kick_rounds:", kick_rounds, ", kick_loss:", kick_loss)

    tour = best_tour("tours")
    score = world.score(tour)
    print(score)

    new_tour, start, stop = kick(tour, world, popmusic)
    new_score = world.score(new_tour)
    print("batch", batch, "gap score:", new_score - score)

    if new_score < score:
        print('Kick successful!!! New_score:', new_score)
        tsp.Tour.save_csv(new_tour, "tours/tour-%.4f.csv" % new_score)
    else: print("Kick failed.")