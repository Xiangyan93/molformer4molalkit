import os
CWD = os.path.dirname(__file__)


AVAILABLE_CHECKPOINTS = []
for fname in os.listdir(CWD):
    if fname.endswith('.ckpt'):
        AVAILABLE_CHECKPOINTS.append(os.path.join(CWD, fname))
