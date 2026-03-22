from setuptools import find_packages, setup

setup(
    name="dswed",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "transformers>=4.36.2",
        "torch>=2.0.0",
        "torchaudio>=0.10.1",
        "joblib>=1.0.1",
        "Levenshtein>=0.27.0",
        "huggingface_hub>=0.20.0",
        "scikit-learn>=1.0.0",
    ],
    author="Yifan Yang",
    author_email="yifanyeung@sjtu.edu.cn",
    description="A package for computing DS-WED.",
    url="https://github.com/yfyeung/DS-WED",
    keywords="speech metrics",
)
