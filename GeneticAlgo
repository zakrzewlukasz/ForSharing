
# from src.model import *
# from src.function import *
# from src.DynamoDB import *

# Plot function
# from cProfile import label
# from matplotlib import pyplot as plt

####################################################################################
# DynamoDB
####################################################################################
import concurrent.futures  # lub to obsługuje wątek proces
from tqdm import tqdm
import pandas as pd
import time
import os
import multiprocessing
import copy
import math
import json
# from symbol import parameters
import boto3
import uuid
from botocore.exceptions import ClientError
import numpy as np

GA_POPULATION_TABLE = 'ga-evo-Records'
RESULTS_TABLE = 'ga-evo-Results'
GA_JAMODEL_H_TABLE = 'JAModelH'
GA_JAMODEL_M_TABLE = 'JAModelM'

dynamodb = boto3.resource('dynamodb')
population_table = dynamodb.Table(GA_POPULATION_TABLE)
result_table = dynamodb.Table(RESULTS_TABLE)
table_JAModelH = dynamodb.Table(GA_JAMODEL_H_TABLE)
table_JAModelM = dynamodb.Table(GA_JAMODEL_M_TABLE)


class DynamoDB_import:
    def __init__(self):
        # load the population, using Set 0 (the only set loaded at this point)
        self.import_population = None
        try:
            response = population_table.get_item(Key={'RecordID': 0})
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            self.import_population = response['Item']['Population']
            print(f'Loaded {len(self.import_population)} gens')
            for gens in self.import_population:
                gens['parameters'] = np.frombuffer(
                    gens['parameters'].value, dtype=np.float64)
                gens['standard_deviation'] = np.frombuffer(
                    gens['standard_deviation'].value, dtype=np.float64)

    def __call__(self, i):
        return self.import_population[i]['parameters'], \
            self.import_population[i]['standard_deviation']


class DynamoDB_JAModel_import:
    def __init__(self):
        # load the JAModel H, using Set 0 (the only set loaded at this point)
        self.import_JAModel_H = None
        self.import_JAModel_M = None
        try:
            response_H = table_JAModelH.get_item(Key={'RecordID': 0})
            response_M = table_JAModelM.get_item(Key={'RecordID': 0})
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            self.import_JAModel_H = response_H['Item']['H_pom']
            self.import_JAModel_M = response_M['Item']['M_pom']
            for _ in range(len(self.import_JAModel_H)):
                self.import_JAModel_H[_] = np.float64(self.import_JAModel_H[_])
            for _ in range(len(self.import_JAModel_M)):
                self.import_JAModel_M[_] = np.float64(self.import_JAModel_M[_])
            # JAModel_H
            # for _ in range(len(self.import_JAModel_H)):
            #     self.import_JAModel_H[_] =  np.frombuffer(self.import_JAModel_H[_].value, dtype=np.float64)

            # #JAModel_M
            # for _ in range(len(self.import_JAModel_M)):
            #     self.import_JAModel_M[_] =  np.frombuffer(self.import_JAModel_M[_].value, dtype=np.float64)

    def __call__(self):
        return self.import_JAModel_H, self.import_JAModel_M


class DynamoDB_export:
    def __init__(self, log, world_size, population_size, migration_interval,
                 migration_size, mutation_rate, max_generations):
        self.best_number = log[0].individual_number
        self.best_score = log[0].score
        self.parameters = log[0].parameters.tolist()
        self.M_model_r = log[0].M_model_r
        # self.M_pom_i = log[0].M_pom_i.tolist()
        self.H_model_r = log[0].H_model_r

        self.all_info = log[1].values
          # self.generation = log[1].values[i][0]
          # self.score = log[1].values[i][1]

        guid = str(uuid.uuid4())
        ddb_data = json.loads('{}')
        ddb_data['GUID'] = guid
        ddb_data['Best_Number'] = self.best_number
        ddb_data['Best_Score'] = str(self.best_score)
        ddb_data['Save_Time'] = str(time.ctime())
        # Wyniki algorytmu co generację
        ddb_data['Values'] = str(log[1].values.tolist())
        ddb_data['Parameters'] = str(self.parameters)
        ddb_data['World_Size'] = str(world_size)
        ddb_data['Population_Size'] = str(population_size)
        ddb_data['Migration_Interval'] = str(migration_interval)
        ddb_data['Migration_Size'] = str(migration_size)
        ddb_data['Mutation_Rate'] = str(mutation_rate)
        ddb_data['Max_Generations'] = str(max_generations)
        ddb_data['M_model_r'] = str(self.M_model_r)
        ddb_data['H_model_r'] = str(self.H_model_r)

        result_table.put_item(Item=ddb_data)


####################################################################################
# Function
####################################################################################


class JilesAtherton:
    def __init__(self, a_model, k_model, Ms_model, c_model, alpha_model):
        self.a_model = a_model
        self.k_model = k_model
        self.Ms_model = Ms_model
        self.c_model = c_model
        self.alpha_model = alpha_model

        self.H_pom = []
        self.M_pom = []

        self.H_model = [0]
        self.M_model = [0]

        self.M_pom_i = []

        db_imported_JAModel = DynamoDB_JAModel_import()

        self.H_pom, self.M_pom = db_imported_JAModel()

        diff = np.diff(self.H_pom)
        diff_ab = [abs(ele) for ele in diff]
        DeltaH = np.average(diff_ab)

        # DeltaH = 3 #20
        Nfirst = 32  # 125
        Ndown = 64  # 250
        Nup = 64  # 250

        for i in range(Nfirst):
            self.H_model.append(self.H_model[i] + DeltaH)

        for i in range(Ndown):
            self.H_model.append(self.H_model[-1] - DeltaH)

        for i in range(Nup):
            self.H_model.append(self.H_model[-1] + DeltaH)

    def __call__(self, parameters):

        # Score function
        score = 0

        a = parameters[0]
        k = parameters[1]
        Ms = parameters[2]
        c = parameters[3]
        # podane w innej skali trzeba dac 10^alpha, aby było ok
        alpha = 10**parameters[4]

        self.M_model = self.function(a, k, Ms, c, alpha)

        # Delete petla pierwotna primary loop reduce H_model and M_model
        H_model_r = self.H_model[32:]
        M_model_r = self.M_model[32:]

        # plt.plot(self.H_pom,self.M_pom, "-og", label = 'H_pom od M_pom')
        # plt.plot(H_model_r,M_model_r, "-ok", label = 'H_model od M_model')

        # Interpolate M Daje te same wartośni niezalenie od parametrów, więc mona to przeniesc do konstruktora klasy?
        M_pom_i1 = np.interp(H_model_r[0:64], np.sort(
            self.H_pom[0:64]), np.sort(self.M_pom[0:64]))
        M_pom_i2 = np.interp(
            H_model_r[64:129], self.H_pom[64:129], self.M_pom[64:129])

        self.M_pom_i = np.concatenate((M_pom_i1, M_pom_i2))

        # Score individual
        scoreV = (self.M_pom_i - M_model_r)**2
        score = np.sum(scoreV)

        # plt.plot(H_model_r, self.M_pom_i, "-ob",  label = 'H model od M(po interpolacji)') # H model
        # plt.legend(loc="upper left")
        # plt.show()

        # minus bo szukamy minimum
        return -math.fabs(score), H_model_r, M_model_r

    def function(self, a, k, Ms, c, alpha):
        delta = [0]
        Man = [0]
        dMirrdH = [0]
        Mirr = [0]
        M = [0]
        # H_pom_tmp = ([0] + self.H_pom)

        for i in range(len(self.H_model) - 1):
            if self.H_model[i + 1] > self.H_model[i]:
                delta.append(1)
            else:
                delta.append(-1)

        for i in range(len(self.H_model) - 1):
            Man.append(Ms * (1 / np.tanh((self.H_model[i + 1] + alpha * M[i]) / a) - a / (
                self.H_model[i + 1] + alpha * M[i])))
            dMirrdH.append((Man[i+1] - M[i]) /
                           (k * delta[i+1] - alpha * (Man[i + 1] - M[i])))
            Mirr.append(Mirr[i] + dMirrdH[i + 1] *
                        (self.H_model[i+1] - self.H_model[i]))
            M.append(c * Man[i + 1] + (1 - c) * Mirr[i + 1])

        return M


####################################################################################
# Model
####################################################################################


class Individual:
    def __init__(self, fitness_function, db_imported, individual_number):
        self.fitness_function = fitness_function
        self.db_imported = db_imported
        self.individual_number = individual_number

        self.parameters, self.standard_deviation = self.data_parameters()
        self.individual_size = self.size()
        self.score, self.H_model_r, self.M_model_r = self.evaluate()
        # self.score = self.evaluate()

    def __str__(self):
        return "score: {},\tdata: {}".format(self.score, self.data)

    def __repr__(self):
        return str(self)

    def size(self):
        return len(self.parameters)

    def evaluate(self):
        return self.fitness_function(self.parameters)

    def reevaluate(self):
        self.score, self.H_model_r, self.M_model_r = self.evaluate()

    def data_parameters(self):
        return self.db_imported(self.individual_number)


class Population:
    def __init__(self, population_size, mutation_rate, fitness_function, db_imported):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.db_imported = db_imported

        assert population_size > 0

        self.individuals = [Individual(
            fitness_function, db_imported, i) for i in range(population_size)]

    def get_best(self):
        self.sort()
        return self.individuals[0]

    def select(self, k):
        # Standaryzacja wag oraz wybranie k osobników. Osobniki słabe duze score \
        #  (z większym prawdobodbieństwem p) mają więszką szansę na wylosowanie
        weights = np.array([x.score for x in self.individuals])
        weights = weights - min(weights) + 1
        weights = weights / weights.sum()

        parents = np.random.choice(self.individuals, size=k, p=weights)

        return parents

    def crossover(self, parent_1, parent_2):
        # Krzyzowanie usredniajace ze współczynnikiem a - a losujemy z U(0,1)
        child_1 = copy.deepcopy(parent_1)
        child_2 = copy.deepcopy(parent_2)

        a = np.random.random_sample()  # losuje od 0 do 1 float
        for i in range(child_1.individual_size):
            child_1.parameters[i] = a * parent_1.parameters[i] + \
                (1 - a) * parent_2.parameters[i]
            child_2.parameters[i] = a * parent_2.parameters[i] + \
                (1 - a) * parent_1.parameters[i]

            child_1.standard_deviation[i] = a * parent_1.standard_deviation[i] + \
                (1 - a) * parent_2.standard_deviation[i]
            child_2.standard_deviation[i] = a * parent_2.standard_deviation[i] + \
                (1 - a) * parent_1.standard_deviation[i]

        return child_1, child_2

    def mutate(self, child):
        # Do sprawdzenia mutation rate
        if np.random.uniform(0, 1) < self.mutation_rate:
            n = child.individual_size
            rand1 = np.random.random_sample()
            tau = 1/((2*n**(1/2))**(1/2))
            tau_prim = 1/((2 * n) ** (1 / 2))
            for i in range(n):
                rand2 = np.random.random_sample()
                child.standard_deviation[i] *= math.exp(
                    tau*rand2 + tau_prim*rand1)
                # if child.standard_deviation[i] < 0.1: # ?????
                #     child.standard_deviation[i] = 0.1
                # rand3 = np.random.uniform(0, child.standard_deviation[i]) # to było
                rand3 = np.random.random_sample()
                child.parameters[i] += rand3*child.standard_deviation[i]

    def sort(self):
        self.individuals = sorted(
            self.individuals, key=lambda x: x.score, reverse=True)
        self.individuals = self.individuals[:self.population_size]

    def run(self):
        self.sort()
        #x = 123451
        #pow(x,x)

        #time.sleep(1)

        parent_1, parent_2 = self.select(2)

        child_1, child_2 = self.crossover(parent_1, parent_2)

        self.mutate(child_1)
        self.mutate(child_2)

        child_1.reevaluate()
        child_2.reevaluate()

        self.individuals.append(child_1)
        self.individuals.append(child_2)


class World:
    def __init__(self,
                 world_size,
                 population_size,
                 mutation_rate,
                 migration_interval,
                 migration_size,
                 fitness_function,
                 db_imported):
        self.world_size = world_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.fitness_function = fitness_function
        self.db_imported = db_imported

        assert world_size > 0
        assert population_size > 0

        self.islands = [Population(population_size, mutation_rate,
                                   fitness_function, db_imported) for i in range(world_size)]

    def migrate(self):
        migrant_groups = []

        for island in self.islands:
            migrant_groups.append({
                "individuals": island.select(self.migration_size),
                "destination": np.random.randint(self.world_size)
            })

        for migrant_group in migrant_groups:
            for individual in migrant_group["individuals"]:
                migrant = copy.deepcopy(individual)
                self.islands[migrant_group["destination"]
                             ].individuals.append(migrant)

    def run_parallel_island(self, island):
        print("X")
        for i in range(self.migration_interval):
            island.run()
            print(i)
        return island

    def run_parallel(self, generations, max_time, target_score):
        assert self.world_size > 1
        assert self.migration_interval > 0
        assert self.migration_size > 0

        log = pd.DataFrame(columns=["generation", "score"])

        splits = generations // self.migration_interval
        status = tqdm(range(splits))
        best_individual = self.islands[0].individuals[0]
        start_time = time.time()

        for split in status:
            with multiprocessing.Pool(processes = self.world_size) as pool:
                self.islands = pool.map(self.run_parallel_island, self.islands)
                

            for island in self.islands:
                if island.get_best().score > best_individual.score:
                    best_individual = island.get_best()

            status.set_description(
                "score: {}".format(best_individual.score))

            log_new_row = pd.DataFrame([{"generation": split * self.migration_interval,
                                        "score": best_individual.score}])
            log = pd.concat([log, log_new_row], ignore_index=True)

            if math.fabs(target_score - best_individual.score) < 1e-32:
                print("Score target reached.")
                return best_individual, log

            if time.time() - start_time >= max_time:
                print("Time limit reached.")
                return best_individual, log

            self.migrate()

        print("Generations limit reached.")
        return best_individual, log

    def run_single_island(self, generations, max_time, target_score):
        assert self.world_size == 1
        assert self.migration_interval == 0
        assert self.migration_size == 0

        log = pd.DataFrame(columns=["generation", "score"])

        status = tqdm(range(generations))
        best_individual = self.islands[0].individuals[0]
        start_time = time.time()

        for generation_idx in status:
            for island in self.islands:
                island.run()

                if island.get_best().score > best_individual.score:
                    best_individual = island.get_best()

            status.set_description("score: {}".format(best_individual.score))

            log_new_row = pd.DataFrame([{"generation": generation_idx,
                                         "score": best_individual.score}])
            log = pd.concat([log, log_new_row], ignore_index=True)

            if math.fabs(target_score - best_individual.score) < 1e-32:
                print("Score target reached.")
                return best_individual, log

            if time.time() - start_time >= max_time:
                print("Time limit reached.")
                return best_individual, log

        print("Generations limit reached.")
        return best_individual, log


####################################################################################
# Genetic algorithm parameters
####################################################################################
WORLD_SIZE = 5
POPULATION_SIZE = 500  # nadane
MIGRATION_INTERVAL = 100  # 0 #100
MIGRATION_SIZE = 5  # 0 #5
#CROSSOVER_RATE = 1.0
#ELITISM_RATE = 0.05
MUTATION_RATE = 0.4  # 0.001 #0.005 lub 0.1
# TOURNEY_SIZE = 3  # czy konieczne?
# MAX_STAGNANT_GENERATIONS = 100  # ?
MAX_GENERATIONS = 5000  # 1000
#FUNCTION_COMPLEXITY = 2
MAX_TIME = 3600  # s
TARGET_SCORE = 0

####################################################################################
# Jiles-Atherton model parameters
####################################################################################
A_MODEL = 470  # A/m
K_MODEL = 483  # A/m
MS_MODEL = 1.48e6  # A/m
C_MODEL = 0.0889
ALPHA_MODEL = 9.38e-4


if __name__ == "__main__":
    db_imported = DynamoDB_import()

    if not os.path.exists("output"):
        os.makedirs("output")

    fitness_function = JilesAtherton(
        A_MODEL, K_MODEL, MS_MODEL, C_MODEL, ALPHA_MODEL)
    world = World(
        WORLD_SIZE,
        POPULATION_SIZE,
        MUTATION_RATE,
        MIGRATION_INTERVAL,
        MIGRATION_SIZE,
        fitness_function,
        db_imported
    )

    log = world.run_parallel(MAX_GENERATIONS, MAX_TIME, TARGET_SCORE)
    #log = world.run_single_island(MAX_GENERATIONS, MAX_TIME, TARGET_SCORE)
    DynamoDB_export(log, WORLD_SIZE, POPULATION_SIZE, MIGRATION_INTERVAL,
                    MIGRATION_SIZE, MUTATION_RATE, MAX_GENERATIONS)
    # best, per_generation_best_scores = find_solution()
    # print(population[0]['parameters'])
    print('')
