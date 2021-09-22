from setuptools import setup, find_packages
from gym_snake import __version__

setup(
    name="gym_snake",
    version=__version__,
    install_requires=["numpy", "gym"],
    packages=find_packages()
)