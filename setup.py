import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hierarchical-bay-cat-encoder",
    version="0.4.5",
    author="Li-Pin Juan",
    author_email="lipin.juan02@gmail.com",
    description="Bayesian target encoder with the capacity to encode classes derived from categorical features that relate to one another in a hierarchical structure.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/libinruan/hierarchical_bayesian_target_encoder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords=['target encoder', 'bayesian', 'bayes', 'categorical features', 'hierarchical']
)