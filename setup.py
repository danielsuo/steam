from setuptools import find_packages, setup
setup(
    name="steam",
    version="0.0.1",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "jupyter",
        "jax",
        "jaxlib"
    ],
    python_requires=">=3.5",
    packages=find_packages()
)
