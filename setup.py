from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmprompt-nexus",
    version="0.1.0",
    author="Ernesto L. Estevanell-Valladares, LLMPromptNexus Contributors",
    author_email="estevanell@etheria.eu",  # Add maintainer email
    description="A unified framework for interacting with Large Language Models (LLMs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EEstevanell/llmprompt-nexus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: CC BY-NC 4.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pyyaml>=6.0",
        "aiohttp>=3.8.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.23.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
)