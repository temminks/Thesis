from PPO import PPO
from models.models import Dense128_128
from project import Project

model = Dense128_128(100)
project_path = "./data/J30/j301_1.sm"
projects = [Project(project_path)]


def test_init():
    ppo = PPO(episodes=1, projects=projects, model=model, action_dim=32)

    assert False
