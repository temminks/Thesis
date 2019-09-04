import re
from itertools import chain, combinations

import pandas as pd

from stochastic import *


class Project:
    """The project class contains the whole project and helper methods. The project
    is parsed from a file in Patterson format and stored as a pandas DataFrame. This
    class is also used to construct a state vector.
    """

    def __init__(self, path, stochastic=None):
        with open(path, "r") as f:
            self.file = f.readlines()

        self.path: str = path
        # self.general_df = self.general()
        self.project_df = self.project()
        self.num_of_tasks = int(self.project_df.loc['jobs (incl. supersource/sink )'])
        self.limits = self.resource_limits()
        self.df = self.init_tasks()

        self.finished_tasks = [1]
        self.running = {}
        self.stochastic = stochastic

    def __repr__(self):
        return "Project({})".format(self.path)

    def __len__(self) -> int:
        return int(self.num_of_tasks)

    def general(self) -> pd.DataFrame:
        """General project information."""
        general = [map(str.strip, line.split(sep=':')) for line in self.file[1:3]]
        return pd.DataFrame(general, columns=['description', 'value'])

    def project(self) -> pd.DataFrame:
        """Project specific data like number of jobs."""
        project = dict(map(str.strip, line.split(sep=':')) for line in self.file[4:7])
        project.update(dict(zip(*[line.split() for line in self.file[13:15]])))
        return pd.DataFrame.from_dict(project, orient='index', columns=['value'])

    def resource_limits(self) -> pd.DataFrame:
        """Calculate the resource constrictions (limits).

        There are always exactly four resources in the test problems.

        :return: A DataFrame with two columns: one for the total amount of
        resources and one for the currently available resources.
        :rtype: pd.DataFrame
        """
        resources = {'R ' + str(i + 1): self.file[-2].split()[i] for i in range(4)}
        resources_df = pd.DataFrame.from_dict(resources, orient='index')
        limits = pd.concat([resources_df, resources_df], axis=1)
        limits.columns = ['total', 'available']

        return limits.apply(pd.to_numeric)

    def extract_tasks(self):
        """Parse the tasks from the Patterson file.

        Blocks are separated by a row of ****

        :return:
        """
        separator = re.compile(r'[*]+')
        dur_sep = re.compile(r'[RND][ \d]{2}|[\w.]+')

        relations = []
        idx = 17  # start index to find the precedence relations
        while not re.findall(separator, self.file[idx]):
            relations.append(self.file[idx])
            idx += 1
        relations_df = pd.DataFrame(columns=relations[0].split())
        for i, rel in enumerate(map(str.split, relations[1:])):
            relations_df.loc[i] = [rel[0], rel[1], rel[2], list(map(int, rel[3:]))]

        durations = []
        idx = idx + 4  # start index to find the task's durations
        header = re.findall(dur_sep, self.file[idx - 2])
        while not re.findall(separator, self.file[idx]):
            durations.append(self.file[idx].strip().split())
            idx += 1
        durations_df = pd.DataFrame(durations, columns=header)

        return relations_df, durations_df

    def init_tasks(self) -> pd.DataFrame:
        """Merge/join relations and durations into one single pd.DataFrame.

        Also: set the column types to numeric (when possible).

        :return: The DataFrame with all tasks, their durations and precedence relations
        :rtype: pd.DataFrame
        """
        relations, durations = self.extract_tasks()
        df = relations.merge(durations, how='right', on='jobnr.')
        df.set_index('jobnr.', inplace=True)

        # find all the successors and add them as a new column to each state
        succ = {'1': []}
        for index, row in df.iterrows():
            for successor in row['successors']:
                if str(successor) in succ:
                    succ[str(successor)].append(int(index))
                else:
                    succ[str(successor)] = [int(index)]
        df['predecessors'] = df.index.map(succ)

        df.index = df.index.map(int)
        return df.apply(pd.to_numeric, errors='ignore')

    def possible_tasks(self, key) -> set:
        """A job can only be started when all of its predecessors are finished.

        This method returns - based on the job with id key and a list of already
        finished jobs - the jobs that could possibly be started. The job should
        neither be already running or finished.

        :param key: a job id, jobs start with id 2 (job 1 is a dummy job)
        :return: a set, i.e. only unique entries, of all the jobs that are
        successors of the input job 'key' that could be started
        :rtype: set
        """
        possible_tasks = [successor for successor in self.df.loc[key].successors
                          if all(predecessor in self.finished_tasks for predecessor
                                 in self.df.loc[successor].predecessors)]

        return set(possible_tasks) - set(self.running.keys()) - set(self.finished_tasks)

    def get_unique_tasks(self) -> list:
        tasks = [self.possible_tasks(job) for job in self.finished_tasks if job not in self.running]
        return list(set(item for sublist in tasks for item in sublist))

    def get_actions(self) -> list:
        """Get all feasible actions, i.e. those actions whose preceding tasks
        are all completed and for which there are enough resources available to
        start the action. An action can consist of no, one and several tasks.

        :return: all feasible actions, i.e. the powerset of all feasible tasks
        """
        unique_tasks = self.get_unique_tasks()
        powerset = chain.from_iterable([list(x) for x in combinations(unique_tasks, r)]
                                       for r in range(1, len(unique_tasks) + 1))

        feasible = [[]]
        feasible.extend([action for action in powerset if self.is_feasible(action)])

        return feasible

    def is_feasible(self, action):
        consumption = self.df[['R 1', 'R 2', 'R 3', 'R 4']].loc[action].sum(axis='rows')
        return min(self.limits.available - consumption) >= 0

    def reset(self):
        """Reset the project (all finished and running tasks and the resource limits)."""
        self.finished_tasks = [1]
        self.running = {}
        self.limits = self.resource_limits()

    def next(self, action, durations):
        """Proceed one step and return next time increment.

        Execute a new action, update the resource limits, and return the time
        increment at which the next task is finished.

        :param action: the action to execute
        :param durations: the stochastic times of the tasks that are started
        :return: the time increment needed until the next task is completed
        """
        # add each task in the action to the running tasks
        for t in action:
            self.running[t] = durations[t - 2]

        self.limits.available = self.limits.available.subtract(
            pd.DataFrame(self.df.loc[key][self.limits.index]
                         for key in action).sum(axis='rows'), fill_value=0)

        if len(self.running) > 0:
            time_next_finished = min(self.running.values())
            for key in list(self.running.keys()):
                self.running[key] -= time_next_finished

                # free resources if the task is finished_tasks and remove the
                # task from the dictionary of running tasks
                if self.running[key] <= 0:
                    self.limits.available = self.limits.available.add(
                        self.df.loc[key][self.limits.index], fill_value=0)
                    self.running.pop(key)
                    self.finished_tasks.append(key)
        else:
            # penalty for waiting although there are no running tasks
            time_next_finished = 1

        return time_next_finished

    def is_finished(self) -> bool:
        return len(self.finished_tasks) >= (self.num_of_tasks - 1)

    def topology(self) -> np.array:
        """Returns the adjacency matrix for the topology.

        The weighted adjacency matrix is 0 if there is no connection between two
        tasks and the duration of the successor if there is a task, i.e. moving
        from a task x to y takes the time of y.

        :return: single column adjacency matrix
        """
        top = []
        for task in self.df.iterrows():
            successors = task[1].successors
            durations = [self.df.loc[s].duration for s in successors]

            row = np.zeros(self.num_of_tasks)
            if len(successors) > 0:
                successors = np.add(successors, -1)
                row[successors] = durations
            top.append(row.tolist())

        return top

    def state(self) -> np.array:
        """Representation of the projects state that consists of the running
        tasks, the durations of the possible successors, the resource needs of
        these successors and the number of successors.

        :return: stacked state representation with dimension (-1, 1)
        :rtype: np.array
        """

        # get a unique list of tasks that could be started
        tasks = [self.possible_tasks(job) for job in self.finished_tasks if job not in self.running]
        possible_tasks = list(set(item for sublist in tasks for item in sublist))

        encoded_durations: list = []
        encoded_resources: list = []
        encoded_successors: list = []
        encoded_running_tasks: list = []

        # starting with task 2: task 1 is merely a dummy task with no information
        for task in range(2, self.num_of_tasks):
            encoded_durations.append(self.df.loc[task].duration if task in possible_tasks else 0)
            encoded_resources.append(list(
                self.df.loc[task][['R 1', 'R 2', 'R 3', 'R 4']])
                                     if task in possible_tasks else [0, 0, 0, 0])
            encoded_successors.append(
                self.df.loc[task]['#successors'] if task in possible_tasks else 0)
            encoded_running_tasks.append(self.running[task] if task in self.running.keys() else 0)

        if self.stochastic:
            encoded_durations = self.get_stochastic_durations(encoded_durations)

        durations, resources, successors, running_tasks = [
            np.reshape(x, (-1, 1)) for x in [
                encoded_durations,
                encoded_resources,
                encoded_successors,
                encoded_running_tasks]
        ]

        return np.concatenate((durations, resources, successors, running_tasks)), \
            np.squeeze(durations)

    def get_stochastic_durations(self, encoded_durations):
        return getattr(Stochastic, self.stochastic)(encoded_durations)
