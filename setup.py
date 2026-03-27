from setuptools import setup, find_packages

setup(
    name="omni-memory",
    version="0.1.0",
    packages=find_packages(include=["msa_memory*", "integrations*", "patches*"]),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "accelerate>=0.25.0",
        "sentencepiece>=0.1.99",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "codex": ["openai>=1.0.0"],
    },
    author="Omni-Memory Contributors",
    description="Production-grade, end-to-end trainable memory system based on MSA.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/omni-memory",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
