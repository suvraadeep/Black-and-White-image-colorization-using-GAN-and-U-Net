from setuptools import find_packages,setup

setup(
    name='B&WGAN',
    version='0.0.1',
    author='suvradeep',
    author_email='dassuvradeep9@gmail.com',
    install_requires=["matplotlib","pandas","numpy","torch","tensorflow"],
    packages=find_packages()
)