# Usage

# normal: python3 main.py --model ../output/neural_net.model --dataset ../gtsrb_all --images ../gtsrb_all/Test --predictions ../predictions_all
# random: python3 main.py --model ../output/random_search.model --optimiser random --dataset ../gtsrb_all --images ../gtsrb_all/Test --predictions ../predictions_all
# hyperband: python3 main.py --model ../output/hyperband.model --optimiser hyperband --dataset ../gtsrb_all --images ../gtsrb_all/Test --predictions ../predictions_all
# bayesian: python3 main.py --model ../output/bayesian.model --optimiser bayesian --dataset ../gtsrb_all --images ../gtsrb_all/Test --predictions ../predictions_all


from arguments import Args
from train import Train_Net
from predict import Predict_Net
from randoms import Random_Search
from bayesian import Bayesian_Optimisation
from hyperband import Hyper_band

if __name__ == '__main__':
	try:
		
		a = Args()
		args = a.parse_arguments()

		# create objects of training and predicting classes
		tr = Train_Net()
		hp = Hyper_band()
		bs = Bayesian_Optimisation()
		rs = Random_Search()
		p = Predict_Net()

		if args["optimiser"] == "bayesian":
			bs.main_train_net()
		elif args["optimiser"] == "hyperband":
			hp.main_train_net()
		elif args["optimiser"] == "random":
			rs.main_train_net()
		else:
			print("[INFO] Training without optimisation")
			tr.main_train_net()
			p.main_predict_net()

	except KeyboardInterrupt:
		pass