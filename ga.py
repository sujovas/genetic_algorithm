import math
import random
import numpy as np
from numpy.random import rand
from numpy.random import randint
import matplotlib.pyplot as plt


# def objective(x):  # rast.m
#      x = np.asarray_chkfinite(x)
#      n = len(x)
#      return 10*n + sum( x**2 - 10 * np.cos( 2 * np.pi * x ))


# def objective(x):  # rosen.m
#     x = np.asarray_chkfinite(x)
#     x0 = x[:-1]
#     x1 = x[1:]
#     return sum((1 - x0) ** 2) + 100 * sum((x1 - x0 ** 2) ** 2)

def objective(x):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * np.sin(np.sqrt( abs( x ))))


# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %f" % (gen, decoded[i], scores[i]))
                sol_all.append(scores[i])
                sol_gen.append(gen)

        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    sol_count.append(len(sol_all)-1)
    return [best, best_eval]


# define range for input
dim = 2
# # bounds rastr
# bounds_single = [-5.12, 5.12]
# # bounds rosenbr
# bounds_single = [-5, 10]
# bounds schw
bounds_single = [-500, 500]

bounds = []
for i in range(dim):
    bounds.append(bounds_single)
# for x in bounds:
#     print(x)

# define total runs
n_runs = 5
# define the total iterations
n_iter = 50
# bits per variable
n_bits = 16
# define the population size
n_pop = 80
# crossover rate
r_cross = 0.8
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
# perform the genetic algorithm search n_runs times
best_sol = [0]
# best_seeds = []
sol_all = []
sol_count = []
sol_gen = []

#run GA for nruns and find best solution
for n in range(n_runs):
    seed = np.random.randint(0,1000000)
    random.seed(seed)
    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    # print('Done!')

    if n == 0:
        best_sol.append(score)
    else:
        non_zero = best_sol[best_sol != 0]
        if score < np.min(non_zero):
            best_sol.append(score)
            # best_seeds.append(seed)
        else:
            best_sol.append(0)
        # best_seeds.append(0)

    decoded = decode(bounds, n_bits, best)
    # print('f(%s) = %f' % (decoded, score))
    # print(best_seeds)

# best_seed  = best_seeds[-1]
# random.seed(best_seed)
# best_sol, best_score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
# best_decoded = decode(bounds, n_bits, best_sol)
# print('f(%s) = %f' % (best_decoded, best_score))
print('sol_all', sol_all)
print('sol_count', sol_count)
# print(len(sol_all))
print('sol_gen', sol_gen)
print('best_sol: ', best_sol)
print('best sol nz', non_zero)

#find position of a best solution
non_zero_last = best_sol[best_sol != 0]
non_zero_sol = np.min(non_zero_last)
print(non_zero_sol)
print(type(non_zero_sol))
best_sol_pos = best_sol.index(non_zero_sol)
# best_sol_pos = best_sol.index(min(non_zero_last))
print('Cislo nejlepsiho reseni: ', best_sol_pos)
sol_up = sol_count[best_sol_pos]
# print('sol_up', sol_up)
sol_low = sol_count[best_sol_pos-1]
# print('sol_low', sol_low)
best_plot_gen = sol_gen[sol_low+1:sol_up]
best_plot_sol = sol_all[sol_low+1:sol_up]
best_plot_gen.append(n_iter)
best_plot_sol.append(best_sol[best_sol_pos])
# print(best_plot_gen)
# print(best_plot_sol)
# best_plot_val =

plt.step(best_plot_gen, best_plot_sol)
plt.grid(axis='both', color='0.95')
plt.title('Best solution for %dD nRUNS = %s, n_iter = %f, n_pop = %g' % (dim, n_runs, n_iter, n_pop))
plt.show()

