from setuptools import setup, find_packages

setup(
    name='reasondrive',
    version='0.1.0',
    description='Reasoning-Enhanced VLMs for Autonomous Driving',
    author='Amirhosein Chahe, Lifeng Zhou',
    author_email='your-email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.36.0',
        'accelerate>=0.25.0',
        'datasets>=2.14.0',
        'pillow>=10.0.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.66.0',
        'wandb>=0.15.0',
        'unsloth>=0.3.5',
    ],
)

