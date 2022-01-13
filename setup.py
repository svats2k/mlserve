from setuptools import setup

setup(
    name="mlserve",
    version="0.1",
    entry_points={
        'console_scripts': [ 
            'mlserve = mlserve.__main__:app'
        ]
    },
    install_requires=[
        'typer',
        'rich',
        'tqdm',
        'msgpack',
        'msgpack_numpy',
        'opencv-python',
        'decord',
        'matplotlib',
        'boto3',
        'ffmpeg-python',
        'requests'
    ]
)