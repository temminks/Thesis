from project import Project

"""
Adjacency list of the first tasks of the 301_1 project
1 - 2 -  3 -  4
2 - 6 - 11 - 13
3 - 7 -  8 - 13
4 - 5 -  9 - 10

Resources of the first tasks
         R1  R2  R3  R4
task 2:   4   0   0   0
task 3:  10   0   0   0
task 4:   0   0   0   3
task 5:   3   0   0   0
task 9:   6   0   0   0
task 10:  0   0   0   1

Available resources:
R1  R2  R3  R4
12  13   4  12

Example:
Execution of tasks (2, 3) is impossible: 4+10>12 (resource 1 constraint)
Execution of tasks (3, 4) is possible: 10+0<12, 0+3<12
"""

project_path = "./data/J30/j301_1.sm"


def test_possible_tasks():
    """Tests for possible tasks at the beginning of the project."""
    project = Project(project_path)
    assert project.possible_tasks(1) == {2, 3, 4}, \
        'task 1 has possible successors, i.e. tasks whose predecessors are finished, {2, 3, 4}'


def test_possible_tasks_1():
    """Tests for possible tasks at the beginning of the project."""
    project = Project(project_path)
    project.finished_tasks += [4]
    assert project.possible_tasks(1) == {2, 3}, \
        'after finishing task 4, it is no longer a possible tasks'


def test_possible_tasks_2():
    """Tests for possible tasks with no possible tasks"""
    project = Project(project_path)
    project.finished_tasks += [2, 3, 4]
    assert project.possible_tasks(1) == set(), \
        'after finishing task 2, 3 and 4, there are no more possible tasks.'


def test_possible_tasks_3():
    """Test for possible tasks."""
    project = Project(project_path)
    project.finished_tasks += [4, 9]
    assert project.possible_tasks(1) == {2, 3}, \
        'after finishing tasks [4, 9], we can start 1\'s successors, except 4'
    assert project.possible_tasks(4) == {5, 10}, 'after finishing' \
                                                 " tasks [4, 9], we can start 4's successors, except 9"


def test_get_actions():
    """Tests for get_actions at the beginning of the project."""
    project = Project(project_path)
    assert project.get_actions() == [[], [2], [3], [4], [2, 4], [3, 4]], \
        'at the beginning, waiting, tasks 2, 3 and 4, as well as combinations' \
        '(2, 4) and (3, 4) are possible'


def test_get_actions_1():
    """Test: add tasks to finished tasks -> tasks 5, 9, 10 could be started, too"""
    project = Project(project_path)
    project.finished_tasks.append(4)
    assert project.get_actions() == [
        [], [2], [3], [5], [9], [10], [2, 5], [2, 9], [2, 10], [3, 10],
        [5, 9], [5, 10], [9, 10], [2, 5, 10], [2, 9, 10], [5, 9, 10]]


def test_get_actions_2():
    """Test: no resources available -> only wait action."""
    project = Project(project_path)
    project.limits['available'] = [0, 0, 0, 0]
    assert project.get_actions() == [[]]


def test_adjacency():
    """"""
    project = Project(project_path)
    assert len(project.topology()) == 32
