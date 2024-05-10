from setuptools import setup, find_packages

setup(
    name='NeuroUtils',
    version='0.1.1',
    description='Library for neural network projects organisation',
    author='Sebastian Borukało',
    author_email='Ciapserr@gmail.com',
    packages=find_packages(),
    install_requires=[
        'opencv-python==4.9.0.80',
        'numpy==1.26.2',
        'scipy==1.11.4',
        'scikit-image==0.22.0',
        'scikit-learn==1.3.2',
        'tensorflow-cpu==2.10.0',
        'tqdm==4.65.0',
        'pandas==2.1.4',
        'matplotlib==3.8.2'
    ],
)
