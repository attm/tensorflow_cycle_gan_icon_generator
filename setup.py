from setuptools import setup


setup(
   name='cycle_gan_icon_generator',
   version='0.1',
   description='Personal project package',
   author='atgm1113',
   author_email='atgm1113@gmail.com',
   packages=["src",
             "src.data_process",
             "src.main",
             "src.models_build",
             "src.tests",
             "src.train",
             "src.model"])