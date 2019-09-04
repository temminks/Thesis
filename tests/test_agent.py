from agents import Agent
from models.models import Dense128_128
from project import Project

model = Dense128_128(100)
project_path = "./data/J30/j301_1.sm"
projects = [Project(project_path)]


def test_agent():
    agent = Agent(projects, model)
    assert agent
