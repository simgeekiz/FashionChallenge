# This code is based on an assignment designed by Paul Valiant.  
# Please do not disseminate this code without his written permission.
#
# This python implementation was written by Okke Schrijvers



import sys
import math
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation
import copy
import random

data = []


# plan is an array of 40 floating point numbers
def sim(plan, save=False):
    for i in range(0, len(plan)):
        if plan[i] > 1:
            plan[i] = 1.0
        elif plan[i] < -1:
            plan[i] = -1.0

    dt = 0.1
    friction = 1.0
    gravity = 0.1
    mass = [30, 10, 5, 10, 5, 10]
    edgel = [0.5, 0.5, 0.5, 0.5, 0.9]
    edgesp = [160.0, 180.0, 160.0, 180.0, 160.0]
    edgef = [8.0, 8.0, 8.0, 8.0, 8.0]
    anglessp = [20.0, 20.0, 10.0, 10.0]
    anglesf = [8.0, 8.0, 4.0, 4.0]
    
    edge = [(0, 1),(1, 2),(0, 3),(3, 4),(0, 5)]
    angles = [(4, 0),(4, 2),(0, 1),(2, 3)]
    
    # vel and pos of the body parts, 0 is hip, 5 is head, others are joints
    v = [[0.0,0.0,0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0,0.0,0.0]]
    p = [[0, 0, -.25, .25, .25, .15], [1, .5, 0, .5, 0, 1.9]]
    
    spin = 0.0
    maxspin = 0.0
    lastang = 0.0
    
    for j in range(len(plan)/2):
        for k in range(10):
            lamb = 0.05 + 0.1*k
            t0 = 0.5
            if j>0:
                t0 = plan[2*j-2]
            t0 *= (1-lamb)
            t0 += plan[2*j]*lamb
            
            t1 = 0.0
            if j>0:
                t1 = plan[2*j-1]
            t1 *= (1-lamb)
            t1 += plan[2*j+1]*lamb
            
            

            contact = [False,False,False,False,False,False]
            for z in range(6):
                if p[1][z] <= 0:
                    contact[z] = True
                    spin = 0
                    p[1][z] = 0

            anglesl = [-(2.8+t0), -(2.8-t0), -(1-t1)*.9, -(1+t1)*.9]

            disp = [[0,0,0,0,0],[0,0,0,0,0]]
            dist = [0,0,0,0,0]
            dispn = [[0,0,0,0,0],[0,0,0,0,0]]
            for z in range(5):
                disp[0][z] = p[0][edge[z][1]]-p[0][edge[z][0]]
                disp[1][z] = p[1][edge[z][1]]-p[1][edge[z][0]]
                dist[z] = math.sqrt(disp[0][z]*disp[0][z] + disp[1][z]*disp[1][z])
                inv = 1.0/dist[z];
                dispn[0][z] = disp[0][z]*inv
                dispn[1][z] = disp[1][z]*inv;
        
            dispv = [[0,0,0,0,0],[0,0,0,0,0]]
            distv = [0,0,0,0,0]
            for z in range(5):
                dispv[0][z] = v[0][edge[z][1]] - v[0][edge[z][0]]
                dispv[1][z] = v[1][edge[z][1]] - v[1][edge[z][0]]
                distv[z] = 2*(disp[0][z]*dispv[0][z] + disp[1][z]*dispv[1][z])

            
            forceedge = [[0,0,0,0,0],[0,0,0,0,0]]
            for z in range(5):
                c = (edgel[z]-dist[z])*edgesp[z]-distv[z]*edgef[z]
                forceedge[0][z] = c*dispn[0][z]
                forceedge[1][z] = c*dispn[1][z]

            edgeang = [0,0,0,0,0]
            edgeangv = [0,0,0,0,0]
            for z in range(5):
                edgeang[z] = math.atan2(disp[1][z], disp[0][z])
                edgeangv[z] = (dispv[0][z]*disp[1][z]-dispv[1][z]*disp[0][z])/(dist[z]*dist[z])

            inc = edgeang[4] - lastang
            if (inc < -math.pi):
                inc += 2.0 * math.pi
            elif inc > math.pi:
                inc -= 2.0 * math.pi
            spin += inc
            spinc = spin - .005*(k + 10 * j)
            if spinc > maxspin:
                maxspin = spinc
                lastang = edgeang[4]

            angv = [0,0,0,0]
            for z in range(4):
                angv[z] = edgeangv[angles[z][1]]-edgeangv[angles[z][0]];

            angf = [0,0,0,0]
            for z in range(4):
                ang = edgeang[angles[z][1]]-edgeang[angles[z][0]]-anglesl[z]
                if ang > math.pi:
                    ang -= 2*math.pi
                elif ang < -math.pi:
                    ang += 2*math.pi
                m0 = dist[angles[z][0]]/edgel[angles[z][0]]
                m1 = dist[angles[z][1]]/edgel[angles[z][1]]
                angf[z] = ang*anglessp[z]-angv[z]*anglesf[z]*min(m0,m1)

            edgetorque = [[0,0,0,0,0],[0,0,0,0,0]]
            for z in range(5):
                inv = 1.0 / (dist[z]*dist[z])
                edgetorque[0][z] = -disp[1][z]*inv
                edgetorque[1][z] =  disp[0][z]*inv

            for z in range(4):
                i0 = angles[z][0]
                i1 = angles[z][1]
                forceedge[0][i0] += angf[z]*edgetorque[0][i0]
                forceedge[1][i0] += angf[z]*edgetorque[1][i0]
                forceedge[0][i1] -= angf[z]*edgetorque[0][i1]
                forceedge[1][i1] -= angf[z]*edgetorque[1][i1]

            f = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            for z in range(5):
                i0 = edge[z][0]
                i1 = edge[z][1]
                f[0][i0] -= forceedge[0][z]
                f[1][i0] -= forceedge[1][z]
                f[0][i1] += forceedge[0][z]
                f[1][i1] += forceedge[1][z]

            for z in range(6):
                f[1][z] -= gravity*mass[z]
                invm = 1.0/mass[z]
                v[0][z] += f[0][z]*dt*invm
                v[1][z] += f[1][z]*dt*invm
                
                if contact[z]:
                    fric = 0.0
                    if v[1][z] < 0.0:
                        fric = -v[1][z]
                        v[1][z] = 0.0

                    s = np.sign(v[0][z])
                    if v[0][z]*s < fric*friction:
                        v[0][z]=0
                    else:
                        v[0][z] -= fric*friction*s
                p[0][z] += v[0][z] * dt
                p[1][z] += v[1][z] * dt;
			
            if save:
			    data.append(copy.deepcopy(p))


            if contact[0] or contact[5]:
                #print("contact exit - 0: {} and 1: {}".format(p[0][5],p[1][5]))
                return p[0][5] #*p[1][5]
	#print("contact exit - 0: {} and 1: {}".format(p[0][5],p[1][5]))
	return p[0][5] #*p[1][5]


###########
# The following code is given as an example to store a video of the run and to display
# the run in a graphics window. You will treat sim(plan) as a black box objective
# function and minimize it.
###########


def random_peeps(number_of_peeps):
	return [[random.uniform(-1,1) for i in range(20)] for _ in range(number_of_peeps)]
	

def crossover(peep1, peep2):
    if(len(peep1) != len(peep2)):
        print("error!!!")
    point = random.randint(0,len(peep1))
    return peep1[0:point] + peep2[point:len(peep2)], peep2[0:point] + peep1[point:len(peep2)]


def mutate(peep, rate=0.05):
	for i in range(len(peep)):
		if random.random() < rate:
			peep[i] = random.uniform(-1, 1)
	return peep


# misschien wel interessant om ook een paar randoms toe te voegen, of maar 1 ofzo
def new_generation(peeps, raw_fitnesses, mutation_rate, crossover_rate=1):
	min_fit = np.min(raw_fitnesses)
	fitnesses = [fit - min_fit for fit in raw_fitnesses]
	sum = np.sum(fitnesses)
	
	chosen_ones = []
	for i in range(len(peeps)-1):
		pick = random.uniform(0, sum)
		idx = 0
		#print('pick:\t' + str(pick))
		while(pick > 0):			
			pick -= fitnesses[idx]
			idx += 1
			#print('pick_{}:\t'.format(idx) + str(pick))
		chosen_ones.append(peeps[idx-1][:])
	chosen_ones = [mutate(one) for one in chosen_ones]
	
	for i in range((len(peeps)-1)/2):
		p1,p2 = crossover(chosen_ones[i*2], chosen_ones[i*2+1])
		chosen_ones[i*2] = p1
		chosen_ones[i*2+1] = p2
		
	i = np.argmax(raw_fitnesses)
	
	chosen_ones.append(peeps[i][:])
	
	return chosen_ones
		

def mainloop(mutation_rate, goal, crossover_rate=1, max_iterations=-1):
	peeps = random_peeps(pop_size)
	fitnesses = [sim(peep + peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]) for peep in peeps]
	count = 0
	while(max(fitnesses) < goal and count != max_iterations):
		peeps = new_generation(peeps, fitnesses, mutation_rate, crossover_rate) 
		fitnesses = [sim(peep + peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]+ peep[10:]) for peep in peeps]
		count += 1
		print(str(count) + ':   ' + str(max(fitnesses)))
	i = np.argmax(fitnesses)
	return peeps[i]
	
	
pop_size = 100 + 1
mutation_rate = 0.1
crossover_rate = 1
goal = 1000
max_iter = 500

from time import time

start = time()
winner = mainloop(mutation_rate, goal, crossover_rate, max_iter)
print(time()-start)

endless_loop = winner + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:]+ winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:] + winner[10:]


print('Final score:\t' + str(sim(endless_loop, True)))



# draw the simulation
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(12, 3)

ax = plt.axes(xlim=(-1, 10), ylim=(0, 3))

joints = [5, 0, 1, 2, 1, 0, 3, 4]
patch = plt.Polygon([[0,0],[0,0]],closed=None, fill=None, edgecolor='k')
head = plt.Circle((0, 0), radius=0.15, fc='k', ec='k')

def init():
    ax.add_patch(patch)
    ax.add_patch(head)
    return patch,head


def animate(j):
    points = zip([data[j][0][i] for i in joints], [data[j][1][i] for i in joints])
    patch.set_xy(points)
    head.center = (data[j][0][5], data[j][1][5])
    return patch,head

anim = animation.FuncAnimation(fig, animate,
                              init_func=init,
                              frames=len(data),
                              interval=20)
#anim.save('animation.mp4', fps=50)

plt.show()