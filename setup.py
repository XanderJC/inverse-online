from setuptools import find_packages, setup

setup(
    name="iol",
    version="0.0.1",
    author="Alex",
    author_email="alexjameschan@gmail.com",
    description="Treatment effect beliefs",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="iol.tests.test_all.suite",
    install_requires=[
        "torch >= 1.8.0",
        "tqdm >= 4.54.1",
        "numpy >= 1.19.1",
        "pandas >= 1.1.2",
        "sklearn >= 0.0",
    ],
)
