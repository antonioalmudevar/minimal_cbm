import argparse

from src.experiments import TestExperiment


def parse_args():

    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument(
        'config_file', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        '-s', '--seed', 
        type=int, 
        default=42, 
        help='seed for initialization'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = TestExperiment(
        **vars(args), wandb_key='ad6dfde6b67458b23b722ca23221f8d82d3cf713')
    experiment.run()