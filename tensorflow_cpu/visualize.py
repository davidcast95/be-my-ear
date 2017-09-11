import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import time




def generate_mesh(arr):
    x = np.arange(0, arr.shape[0], 1)
    y = np.arange(0, arr.shape[1], 1)
    return x,y,arr.T

if (len(sys.argv) < 5):
    print("this method need 2 arguments")
    print("CHECKPOINT_DIR ~> directory of your model's checkpoint")
    print("VISUALIZE_TYPE ~> w for weight, b for biases :{w,b,wb}")
    print("DELAY_TIME ~> int of time in second(s)")
else:
    checkpoint_dir = sys.argv[1]
    visualize_data = sys.argv[2]
    mode = sys.argv[3]
    delay = int(sys.argv[4])
    first = True

    plt.ion()
    if visualize_data == "w":
        fig_w1 = plt.figure("weight 1")
        fig_w2 = plt.figure("weight 2")
        fig_w3 = plt.figure("weight 3")
        fig_w5 = plt.figure("weight 5")
        fig_w6 = plt.figure("weight 6")
        ax_w1 = fig_w1.add_subplot(111)
        ax_w2 = fig_w2.add_subplot(111)
        ax_w3 = fig_w3.add_subplot(111)
        ax_w5 = fig_w5.add_subplot(111)
        ax_w6 = fig_w6.add_subplot(111)
        for root, dirs, files in os.walk(checkpoint_dir, topdown=True):
            for dir in dirs:
                target_dir = os.path.join(checkpoint_dir, dir)
                if mode=="seq":
                    w1 = np.load(os.path.join(target_dir,"w1.npy"))
                    x, y, z = generate_mesh(w1)
                    ax_w1.pcolormesh(x,y,z,cmap="inferno")
                    ax_w1.set_title(dir)
                    fig_w1.canvas.draw()

                    w2 = np.load(os.path.join(target_dir,"w2.npy"))
                    x, y, z = generate_mesh(w2)
                    ax_w2.pcolormesh(x,y,z,cmap="inferno")
                    ax_w2.set_title(dir)
                    fig_w2.canvas.draw()


                    w3 = np.load(os.path.join(target_dir,"w3.npy"))
                    x, y, z = generate_mesh(w3)
                    ax_w3.pcolormesh(x,y,z,cmap="inferno")
                    ax_w3.set_title(dir)
                    fig_w3.canvas.draw()


                    w5 = np.load(os.path.join(target_dir,"w5.npy"))
                    x, y, z = generate_mesh(w5)
                    ax_w5.pcolormesh(x,y,z,cmap="inferno")
                    ax_w5.set_title(dir)
                    fig_w5.canvas.draw()

                    w6 = np.load(os.path.join(target_dir,"w6.npy"))
                    x, y, z = generate_mesh(w6)
                    ax_w6.pcolormesh(x,y,z,cmap="inferno")
                    ax_w6.set_title(dir)
                    fig_w6.canvas.draw()

                    time.sleep(delay)
                elif mode == "diff":
                    w1 = np.load(os.path.join(target_dir,"w1.npy"))
                    w2 = np.load(os.path.join(target_dir,"w2.npy"))
                    w3 = np.load(os.path.join(target_dir,"w3.npy"))
                    w5 = np.load(os.path.join(target_dir,"w5.npy"))
                    w6 = np.load(os.path.join(target_dir,"w6.npy"))
                    if first:
                        old_w1 = w1
                        old_w2 = w2
                        old_w3 = w3
                        old_w5 = w5
                        old_w6 = w6
                        first=False
                    else:
                        diff_w1 = w1-old_w1
                        old_w1 = w1
                        x,y,z = generate_mesh(diff_w1)
                        ax_w1.pcolormesh(x,y,z,cmap="inferno")
                        ax_w1.set_title(dir)
                        fig_w1.canvas.draw()

                        diff_w2 = w2-old_w2
                        old_w2 = w2
                        x,y,z = generate_mesh(diff_w2)
                        ax_w2.pcolormesh(x,y,z,cmap="inferno")
                        ax_w2.set_title(dir)
                        fig_w2.canvas.draw()

                        diff_w3 = w3 - old_w3
                        old_w3 = w3
                        x, y, z = generate_mesh(diff_w3)
                        ax_w3.pcolormesh(x, y, z, cmap="inferno")
                        ax_w3.set_title(dir)
                        fig_w3.canvas.draw()

                        diff_w5 = w5 - old_w5
                        old_w5 = w5
                        x, y, z = generate_mesh(diff_w5)
                        ax_w5.pcolormesh(x, y, z, cmap="inferno")
                        ax_w5.set_title(dir)
                        fig_w5.canvas.draw()

                        diff_w6 = w6 - old_w6
                        old_w6 = w6
                        x, y, z = generate_mesh(diff_w6)
                        ax_w6.pcolormesh(x, y, z, cmap="inferno")
                        ax_w6.set_title(dir)
                        fig_w6.canvas.draw()

                        time.sleep(delay)
