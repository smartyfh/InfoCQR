import json
import random
random.seed(42)

if __name__ == "__main__":
    split = "dev"
    data = json.load(open(f"datasets/qrecc/{split}-modified.json", "r"))
    print(f"number of samples before sampling: {len(data)}")
    
    num_samples = 2000
    sampled_data = random.sample(data, num_samples)
    print(f"number of samples after sampling: {len(sampled_data)}")
    
    cnt = num_samples // 1000
    with open(f"datasets/qrecc/{split}-sampled{cnt}k-modified.json", 'w') as out:
        json.dump(sampled_data, out, indent=2)
    