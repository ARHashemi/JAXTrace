"""
Setup script for JAXTrace package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="jaxtrace",
    version="0.1.0",
    author="JAXTrace Development Team",
    author_email="contact@jaxtrace.org",
    description="Memory-Optimized Particle Tracking with JAX",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jaxtrace/jaxtrace",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": ["jax[cuda]", "pynvml"],
        "interactive": ["plotly>=5.0.0", "jupyter", "ipywidgets"],
        "dev": ["pytest", "black", "flake8", "sphinx"],
        "all": [
            "jax[cuda]", "pynvml", "plotly>=5.0.0", 
            "jupyter", "ipywidgets", "pytest", "black", "flake8", "sphinx"
        ]
    },
    keywords="particle tracking, computational fluid dynamics, jax, gpu, memory optimization",
    project_urls={
        "Bug Reports": "https://github.com/jaxtrace/jaxtrace/issues",
        "Source": "https://github.com/jaxtrace/jaxtrace",
        "Documentation": "https://jaxtrace.readthedocs.io/",
    },
    include_package_data=True,
    package_data={
        "jaxtrace": ["*.py"],
    },
)
