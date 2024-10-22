# pyPlumbing

**pyPlumbing** is a SageMath package that provides specialized plumbing computation tools utilizing SageMath's powerful mathematical capabilities for an efficient computation of Ẑ-invariants. This project is a work in progress. The package is not finished and it's still rough. Any suggestions are welcome.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Installing SageMath (Preferred with Conda)](#installing-sagemath-preferred-with-conda)
  - [Installing pyPlumbing](#installing-pyplumbing)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Efficient computations of Ẑ invariants on _negative_ plumbed manifolds.
- Computation of Ẑ invariants with any simply laced Lie Group.
- Integration with SageMath's mathematical environment.
- Extensible and open-source.

## Prerequisites

- **SageMath**: Version 9.0 or higher.
- **Conda**: Recommended for managing environments and packages.

## Installation

### Installing SageMath (Preferred with Conda)

The preferred method to install SageMath is using Conda, which simplifies environment management and package installation.

1. **Install Conda**

   If you don't have Conda installed, download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create a Conda Environment for SageMath**

   Open your terminal and run:

   ```bash
   conda create -n sage sage python=3.9 -c conda-forge
   ```

   This command creates a new environment named `sage` with SageMath and Python 3.9 installed from the `conda-forge` channel.

3. **Activate the SageMath Environment**

   ```bash
   conda activate sage
   ```

4. **Verify SageMath Installation**

   Run SageMath to ensure it's installed correctly:

   ```bash
   sage
   ```

   You should see the SageMath command-line interface start.

**Additional Resources:**

- SageMath Conda Package: [https://anaconda.org/conda-forge/sage](https://anaconda.org/conda-forge/sage)
- SageMath Installation Guide: [https://doc.sagemath.org/html/en/installation/conda.html](https://doc.sagemath.org/html/en/installation/conda.html)

### Installing pyPlumbing

With SageMath installed and the environment activated, you can now install `pyPlumbing`.

1. **Clone the pyPlumbing Repository**

   ```bash
   git clone https://github.com/yourusername/pyPlumbing.git
   ```

2. **Navigate to the pyPlumbing Directory**

   ```bash
   cd pyPlumbing
   ```

3. **Clean Previous Builds (Optional but Recommended)**

   Ensure that any previous build artifacts are removed:

   ```bash
   rm -rf build/ dist/ src/pyPlumbing/*.c src/pyPlumbing/*.so
   find src/pyPlumbing/ -type f -name '*.py' ! -name '__init__.py' -delete
   ```

4. **Install pyPlumbing**

   Use SageMath's `pip` to install the package without build isolation:

   ```bash
   sage -pip install --no-build-isolation .
   ```

   **Important:** The `--no-build-isolation` flag is necessary because `pyPlumbing` depends on SageMath during the build process, and SageMath isn't available in the isolated build environment that `pip` uses by default.

5. **Verify Installation**

   Start SageMath and import `pyPlumbing`:

   ```bash
   sage
   ```

   In the SageMath shell:

   ```python
   sage: import pyPlumbing
   sage: P = pyPlumbing.Plumbing.from_Seifert_data([1,-1/2,-1/3,-1/5])
   sage: P.display()
   ```

## Usage

After installation, you can use `pyPlumbing` within SageMath:

```python
sage: from pyPlumbing import Plumbing
sage: …
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name.
3. Make your changes and commit them with clear messages.
4. Push your changes to your fork.
5. Submit a pull request describing your changes.

## License

This project is licensed under the unlicense - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author:** Davide Passaro
- **Email:** [passaro.davide@protonmail.com](mailto:passaro.davide@protonmail.com)
- **GitHub:** [d-passaro](https://github.com/d-passaro)
