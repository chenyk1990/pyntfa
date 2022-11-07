import pyntfa as ntfa
import numpy as np
din=np.random.randn(1001)
dout=ntfa.tf(din)

err=np.linalg.norm(din-dout)
print(err)


