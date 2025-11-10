import random
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass

@dataclass
class Solution:
    gene:str
    fitness:float | None




def tournament_selection(solutions_w_fitness:list[Solution], tournament_size:int):
    mating_pool = []
    for _ in solutions_w_fitness:
        best = Solution("None", 0)
        for _ in range(tournament_size):
            curr_element: list[Solution] = random.choices(solutions_w_fitness)
            if curr_element[0].fitness > best.fitness:
                best = curr_element

    return mating_pool

def uniform_crossover(parent_a:Solution, parent_b:Solution, mutation_chance:float = 0.05):
    genes = ['A', 'B']
    mask = # TODO implementing the mask generation method, should generate a mask and its mirror one.

    # TODO Remember to add mutation here




class GeneticAlgorithm:

    def compute_fitness(solution:str) -> int:
        features = parse_solution(solution)
        # TODO grab the features to be used and filter the main df

        neigh = KNeighborsClassifier()
        neigh.fit(X, y)




if __name__ == "__main__":
    print("Hello, World!")