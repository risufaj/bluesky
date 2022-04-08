from multiprocessing.dummy import Process
import sys
import numpy as np
from sympy import re
import bluesky as bs
from bluesky import traffic as tr
from bluesky import settings
from bluesky.stack.simstack import process
from bluesky.traffic.route import Route
from bluesky.navdatabase import Navdatabase
from bluesky.simulation import Simulation
from bluesky.traffic.performance.perfbase import PerfBase
import matplotlib.pyplot as plt
from bluesky.tools import geo
import random
from sacred import Experiment
import networkx as nx
from multiprocessing import Lock, Process, Queue, current_process,cpu_count
import queue
import time

def prog_bar(total, progress):
    
    barLength, status = 50, ""
    num_done = progress
    progress = float(progress) / float(total)
    if progress > 1.:
        return
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
 


def worker(n_runs,task_queue, done):
    bs.init('sim-detached')
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            
            break
        else:
            answer = task()
            done.put(answer)
            #prog_bar(n_runs, done.qsize())
            time.sleep(.5)
            
            


class Simulation(object):
    def __init__(self,center, radius, n_ac, sim_time, n_sources, rpz,num_sim) -> None:
        self.center = center
        self.radius = radius
        self.n_ac = n_ac
        self.sim_time = sim_time
        self.n_sources = n_sources
        self.rpz = rpz
        self.num_sim = num_sim
    def __call__(self):
        sources_position = create_sources(self.center, self.radius, self.n_sources)
        spawn_ac(sources_position, self.radius, self.center, number_of_aircrafts=self.n_sources)
        
        #plot_at(center, radius, sources_position)
        """ Main loop """
        
        
        t = 0
        while bs.sim.simt <=self.sim_time:
            
            if not bs.sim.ffmode:
                change_ffmode()
            simstep()
            t = bs.sim.simt
        
        bs.sim.reset()
        return self.num_sim


    def __str__(self) -> str:
        return f"Run number {self.num_sim}"



ex = Experiment("test-experiment")


def check_boundaries(traf, center, radius):
    """
    Check if any aircraft is out of the scenario bounds. It deletes it if so.
    """
    radius = radius * 1852 # From nm to meters
    id_to_delete = []
    for i in range(traf.ntraf):
        if geo.latlondist(traf.lat[i], traf.lon[i] , center[0], center[1]) > radius:
            id_to_delete.append(traf.id[i])

    if id_to_delete:
        for idx in id_to_delete:
            traf.delete(bs.traf.id.index(idx))


def create_sources(center, radius, n_sources):
    """
    The sources will create a polygon centered in the simulation. The function returns
    a list with each source coordinates
    """
    sources_positions = []

    dist_to_center = radius

    for i in range(n_sources):
        alpha = 360/n_sources * i
        lat, lon = geo.qdrpos(center[0], center[1], alpha, dist_to_center)
        sources_positions.append([lat, lon])

    return sources_positions

def simstep():
    bs.sim.step()
    bs.net.step()
    

def change_ffmode(mode=True):
        bs.sim.ffmode = mode
        bs.sim.dtmult = 50.0


def goal_state(source_pos, radius,center):
    distance_to_goal = radius * np.sqrt(random.random())
    
    angle = random.random() * 360.
    goal_lat, goal_lon = geo.qdrpos(center[0], center[1], angle, distance_to_goal/2)
    return goal_lat, goal_lon

def spawn_ac(sources_position, radius, center, number_of_aircrafts):

    chosen = np.array(len(sources_position)* [False])
    
    n_ac = 0
    while n_ac < number_of_aircrafts:
    #for _ in range(number_of_aircrafts):
        source = random.choice(sources_position)
        source_idx = sources_position.index(source)
        if chosen[source_idx] == False:
            chosen[source_idx] = True
            angle = geo.qdrdist(source[0], source[1], center[0], center[1])[0]
            limit_angle = 45

            acid = str(random.getrandbits(32))
            heading = random.uniform(angle - limit_angle, angle + limit_angle)
            goal_lat, goal_lon = goal_state(source,radius,center)


            bs.traf.cre(acid, actype="M200", aclat=source[0], aclon=source[1], acspd=40, achdg=heading,acgoal_lat=goal_lat, acgoal_lon=goal_lon)
            n_ac += 1


@ex.config
def cfg():
    
    center = (47, 9)
    radius = 1
    n_ac = 5
    sim_time = 1*60
    n_sources = 10
    n_runs = 40
    rpz = 5

@ex.automain
def complexity_simulation(_run, center, radius, n_ac, sim_time, n_sources, n_runs, rpz):

    runs_to_do = Queue()
    runs_done = Queue()
    procs = []
    res = []
    
    
   
    n_runs += 1 #queue.get_nowait() ends prematurely, need to figure out why
    for i in range(n_runs):
        runs_to_do.put(Simulation(center, radius, n_ac, sim_time, n_sources, rpz,i))

    num_workers = cpu_count()
    print(f"{num_workers} PROCESSES ARE BEING CREATED")
    for i in range(num_workers):
        p = Process(target=worker,args=(n_runs,runs_to_do,runs_done))
        procs.append(p)
        p.start()

    for p in procs:
        print("JOINING")
        p.join()
    
    while not runs_done.empty():
        res.append(runs_done.get())
    print(res)
    
    