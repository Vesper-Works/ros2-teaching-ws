from setuptools import find_packages, setup

package_name = 'boykisser'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lcas',
    maintainer_email='student@socstech.support',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'roamer = boykisser.roamer:main',
                'opencv = boykisser.opencv_bridge:main',
                'countour = boykisser.colour_contours:main',
                'square = boykisser.square:main'
        ],
    },
)
