from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="cple",
    version="0.2.0",
    description="Communication Process Lightweight Evaluation runtime platform",
    long_description=README,
    long_description_content_type="text/markdown",
    author="CPLE contributors",
    license="MIT",
    url="https://github.com/zcluu/CPLE",
    project_urls={
        "Source": "https://github.com/zcluu/CPLE",
        "Issues": "https://github.com/zcluu/CPLE/issues",
    },
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "PyYAML",
        "sionna",
    ],
    extras_require={
        "dev": ["pytest", "build", "twine"],
    },
    entry_points={
        "console_scripts": [
            "cple-validate-scenarios=cple.tools.validate_scenarios:main",
        ],
    },
)
