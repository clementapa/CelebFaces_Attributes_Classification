
from hparams import Parameters
from utils.boundary_creator import BoundaryCreator

def main():
    config = Parameters.parse()
    config = config.svm_params
    boundaryCreator = BoundaryCreator(config)
    boundaryCreator.create_boundaries()


if __name__ == "__main__":
    main()


    