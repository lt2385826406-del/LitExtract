from setuptools import setup, find_packages

setup(
    name='litextract',
    version='1.0.0',
    description='Automated Literature Knowledge Extraction for Materials Science',
    author='Materials Informatics Lab',
    packages=find_packages(where='src') + find_packages(where='.', include=['scripts']),
    package_dir={'': 'src', 'scripts': 'scripts'},
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.23.0',
        'pandas>=1.0',
        'pillow>=7.1.2',
        'pyyaml>=5.3.1',
        'requests>=2.23.0',
        'pymupdf>=1.26.0',
        'pdf2image>=1.16.0',
        'networkx>=3.0',
        'pyvis>=0.3.0',
        'gradio>=3.50.2',
    ],
    extras_require={
        'vision': [
            'ultralytics>=8.0.0',
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'opencv-python>=4.6.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
        ],
        'llm': [
            'transformers>=4.40.0',
            'peft>=0.10.0',
            'bitsandbytes>=0.43.0',
            'accelerate>=0.30.0',
            'datasets>=2.18.0',
            'trl>=0.8.0',
        ],
        'ocr': ['pytesseract>=0.3.10'],
        'dev': ['pytest>=7.0', 'ruff>=0.4.0'],
    },
    entry_points={
        'console_scripts': [
            'litextract=scripts.run_pipeline:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Materials Science',
    ],
)
