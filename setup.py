"""
Setup script for ML Trading System
"""
from setuptools import setup, find_packages

setup(
    name="ml-trading-system",
    version="0.1.0",
    description="Automated ML Trading System for Robinhood",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "yfinance>=0.1.70",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "robin-stocks>=2.1.0",
        "ta>=0.10.0",
        "pytest>=6.0.0",
        "pytest-cov>=3.0.0",
        "pytest-asyncio>=0.20.0",
        "pytest-mock>=3.0.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ]
    },
    package_dir={"": "."},
    include_package_data=True,
)