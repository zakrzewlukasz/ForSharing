
    def run_parallel(self, generations, max_time, target_score):
        assert self.world_size > 1
        assert self.migration_interval > 0
        assert self.migration_size > 0

        log = pd.DataFrame(columns=["generation", "score"])

        splits = generations // self.migration_interval
        status = tqdm(range(splits))
        best_individual = self.islands[0].individuals[0]
        start_time = time.time()
        array_parallel = []
        array_scores = []
  
        H_pom = self.islands[0].individuals[0].fitness_function.H_pom
        M_pom = self.islands[0].individuals[0].fitness_function.M_pom
        H_model = self.islands[0].individuals[0].fitness_function.H_model

        processes = []

        for island in self.islands:
            array_parallel.append(population_to_array(island))
            array_scores.append(scores_to_array(island))
                


        for split in status:


            item =[]

            with multiprocessing.Manager() as manager:
                shared_queue = manager.Queue()
                processes = [Process(target=run_parallel_island, args=(array_parallel[_],array_scores[_], H_pom, M_pom, H_model, shared_queue)) for _ in range(len(array_parallel))]
                for process in processes:
                    process.start()
                # read data from the queue
                for _ in range(len(array_parallel)):
                    # get an item from the queue
                    #while not queue.empty():
                    item.append(shared_queue.get())
            
            #This is write new generation into array_pararel 
            array_parallel =[]
            array_scores = []
            for i in range(self.world_size):
                array_parallel.append(item[i][0])
                array_scores.append(item[i][1])   
 

            for i in range(len(array_parallel)):
                array_parallel[i], array_scores[i] =  sort(array_parallel[i], array_scores[i])
            
            best = array_scores[0][0]
            for i in range(len(array_scores)):
                for j in range(len(array_scores[i])):
                    if best < array_scores[i][j]:
                        best = array_scores[i][j]

            #print(best)


            status.set_description(
                "score: {}".format(best))

            log_new_row = pd.DataFrame([{"generation": split * self.migration_interval,
                                        "score": best}])
            log = pd.concat([log, log_new_row], ignore_index=True)

            if math.fabs(target_score - best) < 1e-32:
                print("Score target reached.")
                self.write_object(array_parallel, array_scores)
                for island in self.islands:
                    if island.get_best().score > best_individual.score:
                        best_individual = island.get_best()
                return best_individual, log, get_time(start_time)

            if time.time() - start_time >= max_time:
                print("Time limit reached.")
                self.write_object(array_parallel, array_scores)
                for island in self.islands:
                    if island.get_best().score > best_individual.score:
                        best_individual = island.get_best()
                return best_individual, log, get_time(start_time)

            array_parallel, array_scores = migrate(array_parallel, array_scores)

        print("Generations limit reached.")
        #Calculate M for best
        self.write_object(array_parallel, array_scores)
        for island in self.islands:
            if island.get_best().score > best_individual.score:
                best_individual = island.get_best()

        return best_individual, log, get_time(start_time)

    def run_single_island(self, generations, max_time, target_score):
        assert self.world_size == 1
        assert self.migration_interval == 0
        assert self.migration_size == 0

        log = pd.DataFrame(columns=["generation", "score"])

        status = tqdm(range(generations))
        best_individual = self.islands[0].individuals[0]
        start_time = time.time()

        array_parallel = []
        array_scores = []
  
        H_pom = self.islands[0].individuals[0].fitness_function.H_pom
        M_pom = self.islands[0].individuals[0].fitness_function.M_pom
        H_model = self.islands[0].individuals[0].fitness_function.H_model


        for island in self.islands:
            array_parallel.append(population_to_array(island))
            array_scores.append(scores_to_array(island))

        for generation_idx in status:
            for i in range(len(array_parallel)):
                array_parallel[i], array_scores[i] = run(array_parallel[i], array_scores[i], H_pom, M_pom, H_model)

            for i in range(len(array_parallel)):
                array_parallel[i], array_scores[i] =  sort(array_parallel[i], array_scores[i])
            
            best = array_scores[0][0]
            for i in range(len(array_scores)):
                for j in range(len(array_scores[i])):
                    if best < array_scores[i][j]:
                        best = array_scores[i][j]

            status.set_description("score: {}".format(best))

            log_new_row = pd.DataFrame([{"generation": generation_idx,
                                         "score": best}])
            log = pd.concat([log, log_new_row], ignore_index=True)


            if math.fabs(target_score - best) < 1e-32:
                print("Score target reached.")
                self.write_object(array_parallel, array_scores)
                for island in self.islands:
                    if island.get_best().score > best_individual.score:
                        best_individual = island.get_best()
                return best_individual, log, get_time(start_time)

            if time.time() - start_time >= max_time:
                print("Time limit reached.")
                self.write_object(array_parallel, array_scores)
                for island in self.islands:
                    if island.get_best().score > best_individual.score:
                        best_individual = island.get_best()
                return best_individual, log, get_time(start_time)

        print("Generations limit reached.")
        #Calculate M for best
        self.write_object(array_parallel, array_scores)
        for island in self.islands:
            if island.get_best().score > best_individual.score:
                best_individual = island.get_best()

        return best_individual, log, get_time(start_time)


def individual_to_array(indiv):
    return np.concatenate((indiv.parameters[:, np.newaxis], indiv.standard_deviation[:, np.newaxis]), axis=1)

def population_to_array(pop):
    individuals_data = [individual_to_array(indiv) for indiv in pop.individuals]
    return np.stack(individuals_data)

def score_to_array(indiv):
    return np.array([indiv.score])

def scores_to_array(pop):
    scores = [score_to_array(indiv) for indiv in pop.individuals]
    return np.concatenate(scores)


def run_parallel_island(array_single, scores, H_pom, M_pom, H_model, shared_queue):

    for i in range(MIGRATION_INTERVAL):
        array_single, scores = run(array_single, scores, H_pom, M_pom, H_model)

    shared_queue.put((array_single,scores))


def run(array_single, scores, H_pom, M_pom, H_model):
    array_single, scores = sort(array_single, scores)

    returned = select(2, array_single, scores)
    parent_1, parent_2 = returned[0]

    child_1, child_2 = crossover(parent_1, parent_2)

    mutate(child_1)
    mutate(child_2)

    score_child_1 = reevaluate(child_1, H_pom, M_pom, H_model)
    score_child_2 = reevaluate(child_2, H_pom, M_pom, H_model)

    #Expand dimesion to fits array_single
    child_1 = np.expand_dims(child_1, axis=0)
    child_2 = np.expand_dims(child_2, axis=0)


    array_single = np.concatenate((array_single, child_1))
    array_single = np.concatenate((array_single, child_2))
   
    scores = np.concatenate((scores, [score_child_1, score_child_2]))

    return array_single, scores

def sort(array_single, scores):
    #Sort 3 tables by scores values

    indices = (-scores).argsort()[:]
    scores_sorted = scores[indices]
    array_single_sorted = array_single[indices]
    
    #Select population size best single
    scores_sorted = scores_sorted[:POPULATION_SIZE]
    array_single_sorted =array_single_sorted[:POPULATION_SIZE]

    return array_single_sorted, scores_sorted

def select(k, array_single, scores):
    # Standaryzacja wag oraz wybranie k osobników. Osobniki słabe duze score \
    #  (z większym prawdobodbieństwem p) mają więszką szansę na wylosowanie
    weights = np.arange(len(scores), 0, -1)
    weights = weights / weights.sum()

    all_indices = np.arange(len(scores))
    indices = np.random.choice(all_indices, size=k, p=weights, replace=False)
    score_chosen = scores[indices]

    array_single_chosen = array_single[indices]

    return array_single_chosen, score_chosen

def crossover(parent_1, parent_2):
    # Krzyzowanie usredniajace ze współczynnikiem a - a losujemy z U(0,1)
    child_1 = copy.deepcopy(parent_1)
    child_2 = copy.deepcopy(parent_2)


    a = np.random.random_sample()  # losuje od 0 do 1 float
    for i in range(len(child_1)):
        child_1[i][0] = a * parent_1[i][0] + \
            (1 - a) * parent_2[i][0]
        child_2[i][0] = a * parent_2[i][0] + \
            (1 - a) * parent_1[i][0]

        child_1[i][1] = a * parent_1[i][1] + \
            (1 - a) * parent_2[i][1]
        child_2[i][1] = a * parent_2[i][1] + \
            (1 - a) * parent_1[i][1]

    return child_1, child_2


def mutate(child):
    # Do sprawdzenia mutation rate
    if np.random.uniform(0, 1) < MUTATION_RATE:
        n = len(child)
        rand1 = np.random.random_sample()
        tau = 1/((2*n**(1/2))**(1/2))
        tau_prim = 1/((2 * n) ** (1 / 2))
        for i in range(n):
            rand2 = np.random.random_sample()
            child[i][1] *= math.exp(
                tau*rand2 + tau_prim*rand1)
            #rand3 = np.random.random_sample()
            child[i][0] += rand2*child[i][1]
