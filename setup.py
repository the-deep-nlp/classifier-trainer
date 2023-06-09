from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name="classifier_trainer",
    # url="https://github.com/the-deep/deep-experiments/tree/test-summarization-lib",
    author="Data Friendly Space",
    author_email="nlp@thedeep.io",
    # Needed to actually package something
    description="A library for training classification models.",
    packages=find_packages(where="src"),  # include all packages under src
    package_dir={"": "src"},
    include_package_data=True,
    # Needed for dependencies
    install_requires=[
        "protobuf>=3.19.5",
        "scikit-learn>=0.24.2",
        # "torch>=1.10.2",
        "tqdm>=4.61.2",
        "pytorch-lightning>=1.3.8",
        "transformers>=4.8.2",
        "torchmetrics>=0.4.1",
        "torchtext>=0.10.0",
        "sentencepiece",
        "pandas>=1.2.4",
        "pyYAML<=5.4.1",
    ],
    # *strongly* suggested for sharing
    version="0.1",
    # The license can be anything you like
    license="MIT",
    # We will also need a readme eventually (there will be a warning)
    long_description=open("README.md").read(),
)
