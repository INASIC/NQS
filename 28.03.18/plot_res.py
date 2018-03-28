import numpy as np
import matplotlib.pyplot as plt

plt.ion()


exact=-1.274549484318e+00*20


while(True):
    plt.clf()
    plt.ylabel('Energy')
    plt.xlabel('Iteration #')
    results=np.loadtxt('t.txt')
    plt.plot(results[:,0],results[:,1],color='red')
    plt.axhline(y=exact, xmin=0, xmax=results[-1:,0], linewidth=2, color = 'k',label='Exact')
    if(len(results[:,0])>400):
        fitx=results[-400:-1,0]
        fity=results[-400:-1,1]
        z=np.polyfit(fitx,fity,deg=0)
        p = np.poly1d(z)
        plt.plot(fitx,p(fitx))

        error=(z[0]-exact)/-exact
        plt.gca().text(0.95, 0.8, 'Relative Error : '+"{:.2e}".format(error),
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=15,transform=plt.gca().transAxes)


    plt.legend(frameon=False)
    plt.pause(0.05)



plt.show()
