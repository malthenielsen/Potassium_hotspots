import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
#  import os
import subprocess

#  def runner():
#  runner()



decays = [19, 29, 39]
for d in decays:
    subprocess.call(f"python response_difussion.py --decay {d} --alpha 0 --center 50 --time 300", shell = True)

decays = [29]

for d in decays:
    subprocess.call(f"python both_difussion.py --decay {d} --alpha 0 --center 5 --time 300", shell = True)

#  for d in decays:
    #  subprocess.call(f"python both_difussion.py --decay {d} --alpha 0 --center 50 --time 300", shell = True)

#  for d in decays:
#      subprocess.call(f"python both_difussion.py --decay {d} --alpha 0 --center 100 --time 300", shell = True)
#
#  alphas = [0, 22.5, 45]
#  for d in alphas:
    #  subprocess.call(f"python frequency_difussion.py --decay 29 --alpha {d} --center 5 --time 300", shell = True)

alphas = [0, 22.5, 45]
for d in alphas:
    subprocess.call(f"python response_difussion.py --decay 29 --alpha {d} --center 5 --time 300", shell = True)

time = [200,300, 400]
for t in time:
    subprocess.call(f"python response_difussion.py --decay 29 --alpha 0 --center 5 --time {t}", shell = True)

#  subprocess.call(f"python response_difussion.py --decay 29 --alpha 0 --center 50 --time 300", shell = True)
