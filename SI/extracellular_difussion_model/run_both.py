import numpy as np
from matplotlib import pyplot as plt
#plt.style.use('K_PAPER')
#  import os
import subprocess
from multiprocessing import Pool

#  def runner():
#  runner()


def run(i):
    if i == 0:
        subprocess.call(f"python3 response_difussion.py --decay {19} --alpha 0 --center 5 --time 300 --initial 0", shell = True)
    elif i == 1:
        subprocess.call(f"python3 response_difussion.py --decay {29} --alpha 0 --center 5 --time 300 --initial 0", shell = True)
    elif i == 2:
        subprocess.call(f"python3 response_difussion.py --decay {39} --alpha 0 --center 5 --time 300 --initial 0", shell = True)
    elif i == 3:
        print('yo')
        #subprocess.call(f"python3 both_difussion.py --decay {29} --alpha 0 --center 5 --time 300", shell = True)
    elif i == 4:
        print('yo')
        #subprocess.call(f"python3 both_difussion.py --decay {29} --alpha 0 --center 50 --time 300", shell = True)
    elif i == 5:
        subprocess.call(f"python3 response_difussion.py --decay {29} --alpha 0 --center 50 --time 300 --initial 0", shell = True)
    elif i == 6:
        subprocess.call(f"python3 frequency_difussion.py --decay 29 --alpha {0} --center 5 --time 300", shell = True)
    elif i == 7:
        subprocess.call(f"python3 frequency_difussion.py --decay 29 --alpha {22.5} --center 5 --time 300", shell = True)
    elif i == 8:
        subprocess.call(f"python3 frequency_difussion.py --decay 29 --alpha {45} --center 5 --time 300", shell = True)
    elif i == 9:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha {0} --center 5 --time 300 --initial 1", shell = True)
    elif i == 10:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha {22.5} --center 5 --time 300 --initial 1", shell = True)
    elif i == 11:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha {45} --center 5 --time 300 --initial 1", shell = True)
    elif i == 12:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha {0} --center 5 --time 300 --initial 0", shell = True)
    elif i == 13:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha {22.5} --center 5 --time 300 --initial 0", shell = True)
    elif i == 14:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha {45} --center 5 --time 300 --initial 0", shell = True)
    elif i == 15:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha 0 --center 5 --time 200 --initial 0", shell = True)
    elif i == 16:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha 0 --center 5 --time 300 --initial 0", shell = True)
    elif i == 17:
        subprocess.call(f"python3 response_difussion.py --decay 29 --alpha 0 --center 5 --time 400 --initial 0", shell = True)


if __name__ == '__main__':
    index = np.arange(12,13,1)
    pool = Pool(1)
    pool.map(run, index)
    pool.close()
    pool.join()

#subprocess.call(f"python3 response_difussion.py --decay 29 --alpha 0 --center 50 --time 300", shell = True)
