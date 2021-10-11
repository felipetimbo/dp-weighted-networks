from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="dp-weighted-networks",
    version="0.0.1",
    author="Felipe T. Brito",
    author_email="timbo.felipe@gmail.com",
    description="A package to release weighted networks with differential privacy",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/felipetimbo/dp-weighted-networks",
    packages=find_packages(),
    license='TODO' # TODO: set license
)