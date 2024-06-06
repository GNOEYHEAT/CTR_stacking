import argparse

def getConfig():
    parser = argparse.ArgumentParser(description="stack")
    parser.add_argument('--scaler', default="standard", type=str)
    parser.add_argument('--cv', default=5, type=int)
    parser.add_argument('--seed', default=826, type=int)
    
    args = parser.parse_args()
    return args, parser