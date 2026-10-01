import numpy as np
from data import make_patterns, make_test_queries
from recovery_model import RecallModel, build_default_R
from metrics import retrieval_accuracy
from adapters.dummy import DummyAgent

class A:
 def __init__(self,X,p,mode): self.X=X; self.N=X.shape[1]; self.mode=mode
 def predict_precision(self,q):
  # likelihood-ish nearest target
  sims=self.X@q; t=self.X[np.argmax(sims)]
  # residual to candidate: observed dims should q~t, masked q~0
  # score probability observed based closeness q to t vs q to 0
  obs=np.exp(-((q-t)/.08)**2); mask=np.exp(-(q/.08)**2); po=obs/(obs+mask+1e-12)
  if self.mode=='lowobs': return 1.5-po
  if self.mode=='highobs': return .5+po
  if self.mode=='invabs': return 1/(np.abs(q)+.03)
  if self.mode=='abs': return np.abs(q)+.03
  if self.mode=='target': return np.abs(t)+.03
  if self.mode=='invtarget': return 1/(np.abs(t)+.03)
  return np.ones(self.N)
for seed in [42,101]:
 X=make_patterns(seed=seed); m=RecallModel(X,build_default_R(seed=seed)); qs,ts,_=make_test_queries(X,[.75,.85],30,seed)
 b=retrieval_accuracy(m,DummyAgent(X,{}),qs,ts); print('seed',seed,'base',b)
 for mode in ['lowobs','highobs','invabs','abs','target','invtarget']:
  v=retrieval_accuracy(m,A(X,{},mode),qs,ts); print(mode,round(v,3),round(v-b,3))
