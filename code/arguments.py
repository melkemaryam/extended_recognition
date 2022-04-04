import argparse


class Args():


	def parse_arguments(self):
		# create argument parser
		ap = argparse.ArgumentParser()
		ap.add_argument("-m", "--model", required=True, help="path to output model")
		ap.add_argument("-op", "--optimiser", choices=['bayesian', 'hyperband', 'random'], required=False, help="optimisation method for classifier")
		ap.add_argument("-d", "--dataset", required=True, help="path to input GTSRB")
		ap.add_argument("-i", "--images", required=True, help="path to testing directory containing images")
		ap.add_argument("-pr", "--predictions", required=True, help="path to the output predictions directory")
		args = vars(ap.parse_args())

		return args