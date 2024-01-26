import sys
import os
import torch.distributed.launch as launcher


if __name__ == '__main__':
    nproc_per_node = os.environ['N_GPU']
    sys.argv.insert(1,'--distributed')
    sys.argv.insert(1,'train_retriever.py')
    sys.argv.insert(1, str(nproc_per_node))
    sys.argv.insert(1,'--nproc_per_node')
    launcher.main()
