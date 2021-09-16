from setuptools import setup, find_packages
from gym_snake import __version__

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="gym_snake",
    version=__version__,
    author="Aleksa Ćuković",
    author_email="aleksacukovic1@gmail.com",
    description="Implementation of a classic Snake game as gym environment",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/AleksaC/gym-snake",
    license="MIT",
    install_requires=["numpy", "gym"],
    packages=find_packages()
)
