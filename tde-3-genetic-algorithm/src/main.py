import random
from dataclasses import dataclass
from itertools import batched

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_STATE = 23
NUM_FEATURES = 30


@dataclass
class Solution:
    gene: str
    fitness: float | None = None

    def __hash__(self):
        return hash(self.gene)

    def __eq__(self, other):
        if not isinstance(other, Solution):
            return NotImplemented # Or raise TypeError
        return self.gene == other.gene


# TODO maybe add elitism for n genes then tournament for pool_size - n
# mating pool will be the same size as the
def tournament_selection(solutions_w_fitness: list[Solution], tournament_size: int):
    mating_pool = []
    for _ in solutions_w_fitness:
        best = Solution("None", 0)
        for _ in range(tournament_size):
            curr_element: list[Solution] = random.choice(solutions_w_fitness)
            if curr_element.fitness > best.fitness:
                best = curr_element
        mating_pool.append(best)

    return mating_pool


def uniform_crossover(
    parent_a: Solution,
    parent_b: Solution,
    mutation_chance: float = 0.05,
    crossover_rate: float = 0.85,
):
    # mutate before anything since the inversion will be kept regardless
    for i in range(NUM_FEATURES):
        if random.uniform(0, 1) <= mutation_chance:
            inverse = "0" if parent_a.gene[i] == "1" else "1"
            parent_a.gene = parent_a.gene[:i] + inverse + parent_a.gene[i + 1 :]
        # need to re-run the flip
        if random.uniform(0, 1) <= mutation_chance:
            inverse = "0" if parent_a.gene[i] == "1" else "1"
            parent_b.gene = parent_b.gene[:i] + inverse + parent_b.gene[i + 1 :]

    if random.uniform(0, 1) > crossover_rate:
        # no crossover
        return parent_a, parent_b

    mask = random.choices(["A", "B"], k=NUM_FEATURES)
    inv_mask = ["B" if k == "A" else "A" for k in mask]
    child1 = []
    child2 = []
    for i in range(NUM_FEATURES):
        if mask[i] == "A":
            child1.append(parent_a.gene[i])
        else:
            child1.append(parent_b.gene[i])
        if inv_mask[i] == "A":
            child2.append(parent_a.gene[i])
        else:
            child2.append(parent_b.gene[i])
    # returns two new childs
    return Solution("".join(child1)), Solution("".join(child2))


class GeneticAlgorithm:
    def __init__(self, x_df, y_df, n_generations: int, initial_population_size: int):
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_df, y_df, test_size=0.4, random_state=RANDOM_STATE
        )
        # I forgot that train_test returns np arrays so I did all the logic assuming they were dfs, wont change it fuck it
        self.x_train = pd.DataFrame(x_train)
        self.y_train = pd.DataFrame(y_train)

        x_test, x_val, y_test, y_val = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
        )
        self.x_test = pd.DataFrame(x_test)
        self.x_val = pd.DataFrame(x_val)
        self.y_test = pd.DataFrame(y_test)
        self.y_val = pd.DataFrame(y_val)

        self.pop_size = initial_population_size
        self.n_generations = n_generations

    def compute_fitness(self, solution: Solution):
        # TODO add check if already computed solution, would only make sense if i create a cache of solutions
        features = []
        for i in range(len(solution.gene)):
            if solution.gene[i] == "1":
                features.append(self.x_train.iloc[:, i].name)
            else:
                pass
        gene_train = self.x_train.loc[:, self.x_train.columns.isin(features)]
        gene_test = self.x_test.loc[:, self.x_test.columns.isin(features)]
        neigh = KNeighborsClassifier()
        neigh.fit(gene_train, self.y_train.to_numpy().ravel())
        gene_pred = neigh.predict(gene_test)

        solution.fitness = accuracy_score(
            y_true=self.y_test, y_pred=gene_pred, normalize=True
        )

    def run(self):
        population = []
        for i in range(self.pop_size):
            solution = Solution("".join(random.choices(["1", "0"], k=NUM_FEATURES)))
            population.append(solution)
        print(f"Initial population created with {len(population)} individuals.")

        for i in range(self.n_generations):
            print(f"Generation {i}")
            curr_best = Solution("", 0)
            for solution in population:
                self.compute_fitness(solution)
                curr_best = solution if solution.fitness > curr_best.fitness else curr_best
            print(f"Best from generation: {curr_best.gene} with {curr_best.fitness} accuracy")
            selected_parents = tournament_selection(population, 100)
            # reset the population
            population = []
            random.shuffle(
                selected_parents
            )  # randomizing parent combinations, since batched goes (1,2)(3,3)...
            for parents in batched(selected_parents, 2):
                children = uniform_crossover(parents[0], parents[1])
                population.extend(children)

        return set(population)


if __name__ == "__main__":
    random.seed(RANDOM_STATE)
    x_df, y_df = load_breast_cancer(as_frame=True, return_X_y=True)

    ga = GeneticAlgorithm(
        x_df.to_numpy(), y_df.to_numpy(), n_generations=100, initial_population_size=5000
    )
    final_solutions = ga.run()
