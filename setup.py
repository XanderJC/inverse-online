from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Alex",
    author_email="alexjameschan@gmail.com",
    description="Treatment effect beliefs",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
