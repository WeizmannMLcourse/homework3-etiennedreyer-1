import os
import sys

from evaluate import evaluate_on_dataset


def test_part1():

	path_to_ds = 'mnist_pointclouds.h5'

	if not os.path.exists(path_to_ds):
		os.system('wget https://www.dropbox.com/s/3vuzm1mqwa08zvq/mnist_pointclouds.h5')

	accuracy = evaluate_on_dataset(path_to_ds)

	assert accuracy > 0.80, "Sorry your accuracy is too low: %f < 0.80" % accuracy


