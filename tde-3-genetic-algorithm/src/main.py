import random
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass

@dataclass
class Solution:
    gene:str
    fitness: float | None = None




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
    mask = random.choices(['A', 'B'], k=30)
    inv_mask = ['B' if k =='A' else 'A' for k in mask ]
    child1 = []
    child2 = []
    for i in range(30):
        if mask[i] == 'A':
            child1.append(parent_a.gene[i])
        else:
            child1.append(parent_b.gene[i])
        if inv_mask[i] == 'A':
            child2.append(parent_a.gene[i])
        else:
            child2.append(parent_b.gene[i])
    # returns two new childs
    return Solution(''.join(child1)), Solution(''.join(child2))

    # TODO Remember to add mutation here




class GeneticAlgorithm:

    def compute_fitness(solution:str) -> int:
        features = parse_solution(solution)
        # TODO grab the features to be used and filter the main df

        neigh = KNeighborsClassifier()
        neigh.fit(X, y)




if __name__ == "__main__":
    print("Hello, World!")