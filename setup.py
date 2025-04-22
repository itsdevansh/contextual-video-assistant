from setuptools import setup, find_packages

setup(
    name="video-assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'gradio>=4.0.0',
        'requests>=2.28.0',
    ],
)