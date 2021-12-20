import bluesky as bs
from bluesky.tools.aero import ft, nm, vcas2mach,vcas2tas
import numpy as np
from logger import *
import os
import random
import copy
from math import *
from bluesky import stack

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def creconfs(latref, lonref, altref, trkref, gsref, hdg, cpa, tlosh, acid2, pzr):
    trkref = radians(trkref)
    # pzr     = 0.216 * aero.nm

    pzh = 0 * ft
    trk = radians(trkref) + radians(hdg)
    # cpa = cpa * aero.nm
    acalt = altref
    acvs = 0.0

    # gsref = gsref * aero.kts #speed needs to be in m/s
    gsref = vcas2tas(gsref, acalt)

    gs = gsref

    # Horizontal relative velocity vector
    gsn, gse = gs * cos(trk), gs * sin(trk)
    vreln, vrele = gsref * cos(trkref) - gsn, gsref * sin(trkref) - gse
    # Relative velocity magnitude
    vrel = sqrt(vreln * vreln + vrele * vrele)

    # Relative travel distance to closest point of approach
    drelcpa = tlosh * vrel + (0 if cpa > pzr else sqrt(pzr * pzr - cpa * cpa))
    # Initial intruder distance
    dist = sqrt(drelcpa * drelcpa + cpa * cpa)

    # Rotation matrix diagonal and cross elements for distance vector
    rd = drelcpa / dist
    rx = cpa / dist
    # Rotate relative velocity vector to obtain intruder bearing
    brn = degrees(atan2(-rx * vreln + rd * vrele, rd * vreln + rx * vrele))
    # Calculate intruder lat/lon
    aclat, aclon = bs.tools.geo.kwikpos(latref, lonref, brn, dist / nm)

    # no wind gs = tas
    tasn, tase = gsn, gse

    acspd = bs.tools.aero.tas2cas(sqrt(tasn * tasn + tase * tase), acalt)  # / aero.kts

    achdg = degrees(atan2(tase, tasn))

    actype = 'M200'
    acid = 'CONFAC'  # need to find a better scheme

    # return actype, acalt, acspd, aclat, aclon, achdg, acid
    return {'type': actype, 'id': acid2, 'lat': aclat, 'lon': aclon, 'spd': acspd, 'alt': acalt, 'hdg': achdg, 'vs': 0}


def conflict_detection(ownship, intruder, rpz, hpz, dtlookahead):
    I = np.eye(len(ownship))

    # Horizontal conflict ------------------------------------------------------

    # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
    qdr, dist = bs.tools.geo.kwikqdrdist_matrix(np.asmatrix(np.array([ac['lat'] for ac in ownship])),
                                       np.asmatrix(np.array([ac['lon'] for ac in ownship])),
                                       np.asmatrix(np.array([ac['lat'] for ac in intruder])),
                                       np.asmatrix(np.array([ac['lon'] for ac in intruder])))

    # Convert back to array to allow element-wise array multiplications later on
    # Convert to meters and add large value to own/own pairs
    qdr = np.asarray(qdr)
    dist = np.asarray(dist) * bs.tools.aero.nm + 1e9 * I

    # Calculate horizontal closest point of approach (CPA)
    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)  # is pos j rel to i
    dy = dist * np.cos(qdrrad)  # is pos j rel to i

    # Ownship track angle and speed
    owntrkrad = np.radians(np.array([ac['hdg'] for ac in ownship]))
    ownu = np.array([ac['spd'] for ac in ownship]) * np.sin(owntrkrad).reshape((1, len(ownship)))  # m/s
    ownv = np.array([ac['spd'] for ac in ownship]) * np.cos(owntrkrad).reshape((1, len(ownship)))  # m/s

    # Intruder track angle and speed
    inttrkrad = np.radians(np.array([ac['hdg'] for ac in intruder]))
    intu = np.array([ac['spd'] for ac in intruder]) * np.sin(inttrkrad).reshape((1, len(ownship)))  # m/s
    intv = np.array([ac['spd'] for ac in intruder]) * np.cos(inttrkrad).reshape((1, len(ownship)))  # m/s

    du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
    dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
    vrel = np.sqrt(dv2)

    tcpa = -(du * dx + dv * dy) / dv2 + 1e9 * I

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

    # Check for horizontal conflict
    R2 = rpz * rpz
    swhorconf = dcpa2 < R2  # conflict or not

    # Calculate times of entering and leaving horizontal conflict
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
    dtinhor = dxinhor / vrel

    tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
    touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf

    # Vertical conflict --------------------------------------------------------

    # Vertical crossing of disk (-dh,+dh)
    dalt = np.array([ac['alt'] for ac in ownship]).reshape((1, len(ownship))) - \
           np.array([ac['alt'] for ac in intruder]).reshape((1, len(ownship))).T + 1e9 * I

    dvs = np.array([ac['vs'] for ac in ownship]).reshape(1, len(ownship)) - \
          np.array([ac['vs'] for ac in intruder]).reshape(1, len(ownship)).T
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero

    # Check for passing through each others zone
    tcrosshi = (dalt + hpz) / -dvs
    tcrosslo = (dalt - hpz) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)

    # Combine vertical and horizontal conflict----------------------------------
    tinconf = np.maximum(tinver, tinhor)
    toutconf = np.minimum(toutver, touthor)

    swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) * \
                       (tinconf < dtlookahead) * (1.0 - I), dtype=np.bool)

    # --------------------------------------------------------------------------
    # Update conflict lists
    # --------------------------------------------------------------------------
    # Ownship conflict flag and max tCPA
    inconf = np.any(swconfl, 1)
    tcpamax = np.max(tcpa * swconfl, 1)

    # Select conflicting pairs: each a/c gets their own record
    confpairs = [(ownship[i]['id'], ownship[j]['id']) for i, j in zip(*np.where(swconfl))]
    swlos = (dist < rpz) * (np.abs(dalt) < hpz)
    lospairs = [(ownship[i]['id'], ownship[j]['id']) for i, j in zip(*np.where(swlos))]

    return confpairs,  # lospairs, inconf, tcpamax, \
    # qdr[swconfl], dist[swconfl], np.sqrt(dcpa2[swconfl]), \
    # tcpa[swconfl], tinconf[swconfl]


def generate(tot_ac, conf_ac, radius, min_speed, max_speed, h_thresh):
    generated_hdgs = []
    cre_ac = []
    TYPE = "M200"
    BCN_LAT = 41.390205
    BCN_LON = 2.154007
    ALT = 150
    spd1 = np.random.randint(min_speed, max_speed + 1)
    hdg1 = np.random.randint(1, 360)
    hdg_groups = [0, 45, 90, 135, -45, -90, -135, 180]
    chosen_groups = []
    variance = 10

    h_thresh_nm = h_thresh / bs.tools.aero.nm

    # initial aircraft
    ac1 = {'type': TYPE, 'id': 'AC1', 'lat': BCN_LAT, 'lon': BCN_LON, 'spd': spd1, 'alt': ALT, 'hdg': hdg1, 'vs': 0}
    generated_hdgs.append(hdg1)
    last_ac = ac1
    cre_ac.append(ac1)
    # conf_sev = np.random.uniform(0.2,1.0)
    # cpa = h_thresh - h_thresh* conf_sev

    for i in range(1, conf_ac):
        done = False
        runs = 0
        while not done:
            # conf_hdg = np.random.randint(0,90)
            if len(chosen_groups) == 8:
                chosen_groups = []
            group = np.random.choice(list(set(hdg_groups) - set(chosen_groups)))
            # conf_hdg = group + np.random.randint(-variance, variance)
            # chosen_groups.append(group)
            conf_hdg = group + np.random.randint(-variance, variance)
            conf_sev = np.random.uniform(0.2, 1.0)
            cpa = h_thresh - h_thresh * conf_sev
            conf_t = 15  # s

            # take a random aircraft from list of created aircraft to induce conflict
            rand_ac = cre_ac[np.random.randint(0, len(cre_ac))]

            # create conflict with refernce aircraft
            ac = creconfs(last_ac["lat"], last_ac["lon"], last_ac["alt"], last_ac["hdg"],
                          last_ac["spd"], conf_hdg, cpa, conf_t, "CONF_" + str(i), h_thresh)
            confpairs = conflict_detection(cre_ac + [ac], cre_ac + [ac], h_thresh, 100, 8)

            if len(confpairs[0]) == 0:
                last_ac = ac
                cre_ac.append(ac)
                generated_hdgs.append(ac['hdg'])
                done = True
                #print("NCONF_" + str(i) + " ju deshen " + str(runs) + " prova")
            else:

                runs += 1

    for i in range(0, tot_ac - conf_ac):
        hdg = np.random.randint(0, 360)
        dist = np.random.uniform(10 * h_thresh_nm, radius)
        lat, lon = bs.tools.geo.kwikpos(BCN_LAT, BCN_LON, hdg, dist)
        spd = np.random.randint(min_speed, max_speed + 1)
        cre_ac.append({"type": TYPE, "id": "NCONF_" + str(i), "lat": lat,
                       "lon": lon, "spd": spd, "alt": ALT, "hdg": hdg})

    t = "00:00:00"
    tr = "00:01:00"
    scenario = ""
    scenario += t + "> circle exp " + " ".join([str(BCN_LAT), str(BCN_LON), str(radius), "\n"])
    for ac in cre_ac:
        scenario += t + ">" + " ".join(["CRE", ac['id'], ac['type'],
                                        str(ac['lat']), str(ac['lon']),
                                        str(int(round(ac['hdg']))),
                                        str(int(round(ac['alt']))),
                                        str(int(round(ac['spd']))), "\n"])
    scenario += t + ">" + "ZONER " + str(h_thresh_nm) + "\n"
    scenario += t + ">" + "DTLOOK 00:00:" + str(8) + "\n"
    scenario += tr + ">reset \n"

    # with open("C:\\Users\\ralvi\\Desktop\\puna\\current-bluesky\\scenario\\test-example.scn","w") as f:
    #        f.write(scenario)

    return cre_ac, scenario, generated_hdgs

def loadscn():
    path = os.getcwd() + "/scenario/rl3"
    ac = 5  # sa drona duam ne konflikt, mund tjete me shum psh [2,3,4,5,6, etj]

    num_ac_gjithsej = 3

    rrezja = 5  # zonen qe duam te studiojm e perkufizojme me rreth, dhe kjo jep rrezen e rrethit.
    # te gjith dronat krijohen brenda rrethit, normalisht sa me i vogel aq me i dendur trafiku. jepet ne nautical miles
    min_speed = 5  # m/s. shpejtesia minimale qe mund t krijohen
    max_speed = 15  # m/s
    h_thresh = 240  # ne metra. distanca e siguris
    lst, skenar,generated_hdgs = generate(num_ac_gjithsej, ac, rrezja, min_speed, max_speed, h_thresh)
    name = "sim" + "_" + str(ac) + "_" + str(0) + ".scn"
    with open(path+'/'+name, "w") as f:  # ky pathi duhet ndryshuar,ama duhen krijuar brenda folderit scenario

        f.write(skenar)

    #scenario = random.choice(os.listdir(path))
    #print(scenario)
    bs.stack.ic(path + '/' + name)
    bs.sim.step()
    bs.net.step()
    return generated_hdgs

class Environment:
    def __init__(self, n_agents,max_agents = 5):
        super(Environment, self).__init__()

        bs.init("sim-detached")
        self.max_agents = max_agents
        self.timestep = 2 #decision making
        self.agents_id_idx = {bs.traf.id[i]: i for i in range(bs.traf.ntraf)} #duhet ndryshu nqs vendosim qe duhet me qen vec ata n konflikt
        self.agents_id = []
        self.scenario = Saved_scenario()
        self.logger = DataLoggerLists(bs.traf.id)
        self.simulation_time = 59 #kohzgjatja e skenarit
        self.agents_step = []
        self.ini_n_planes = n_agents
        self.conf_ac = []
        self.t_last_solved = 0
        self.first_t_loss = -1
        #self.number_of_planes = 5

        self.close_lat = np.zeros(self.ini_n_planes, dtype=np.float32)
        self.close_lon = np.zeros(self.ini_n_planes, dtype=np.float32)
        self.close_alt = np.zeros(self.ini_n_planes, dtype=np.float32)
        self.close_spd = np.zeros(self.ini_n_planes, dtype=np.float32)
        self.close_hdg = np.zeros(self.ini_n_planes, dtype=np.float32)
        self.distinm = np.zeros(self.ini_n_planes, dtype=np.float32)
        self.qdr = np.zeros(self.ini_n_planes, dtype=np.float32)

        self.reward_ind = np.zeros((self.ini_n_planes, 8), dtype=np.float32) # 8 do zevendsohet me sa faktor t reward do kemi
        self.distinm_diff = np.zeros(self.ini_n_planes, dtype=np.float32)

    def save_close_values(self, ac, minindex, distinm, qdr):
        acid = bs.traf.id[ac]
        self.close_lat[ac] = self.logger.dataloggers[acid].lat[minindex]
        self.close_lon[ac] = self.logger.dataloggers[acid].lon[minindex]
        self.close_alt[ac] = self.logger.dataloggers[acid].alt[minindex]
        self.close_spd[ac] = self.logger.dataloggers[acid].cas[minindex]
        self.close_hdg[ac] = self.logger.dataloggers[acid].hdg[minindex]
        self.distinm[ac] = distinm
        self.qdr[ac] = qdr

    def update_agents(self):
        print("updating")
        self.conf_ac = []

        for confpair in bs.traf.cd.confpairs_unique:
            for ac in confpair:
                if ac not in self.conf_ac:
                    self.conf_ac.append(ac)
        print(self.conf_ac)
        if len(self.conf_ac) == 0:
            self.t_last_solved += 1
        else:
            self.t_last_solved = 0


    def reward(self):

        hdg_max = 90
        rewards = np.zeros(shape=(self.ini_n_planes), dtype=np.float32)
        for agent_idx, acid in enumerate(bs.traf.id):
            reward = 0
            ac = bs.traf.id2idx(acid)
            if acid in self.conf_ac:
                hdg_diff = abs(bs.traf.hdg[ac] - self.close_hdg[ac])
                #offtrack = (1 - (hdg_diff/90)) if hdg_diff < hdg_max else -1
                offtrack = hdg_diff/90 if hdg_diff < hdg_max else 1
                reward -= offtrack

                n_conflicts = sum(bs.traf.id[ac] in conf for conf in bs.traf.cd.confpairs_unique)
                reward -= n_conflicts
            conf_dist = []
            for idx, confpair in enumerate(bs.traf.cd.confpairs):
                if confpair[0] == bs.traf.id[ac]:

                    x = bs.traf.cd.dcpa[idx] / bs.traf.cd.rpz
                    if x == 0:
                        x = 0.01
                    conf = 1 - np.exp(1- 1/x**0.5)
                    conf_dist.append(conf)
            conf_reward = max(conf_dist) if len(conf_dist) > 0 else 0
            reward -= conf_reward

            rewards[ac] = reward


        return rewards

    def act(self,actions):
        for idx, action in enumerate(actions):
            if bs.traf.id[idx] not in self.conf_ac:
                print("Recommended action for "+bs.traf.id[idx]+" is "+str(action)+" however it's not in a conflict so it must not take an action")
            else:
                if action == 1:
                    #print("Action is turn right by 15 degree")
                    bs.traf.ap.selhdgcmd(idx, bs.traf.hdg[idx] + 15)
                elif action == 2:
                    #print("Action is turn left by 15 degree")
                    bs.traf.ap.selhdgcmd(idx, bs.traf.hdg[idx] - 15)
                else:
                    noaction = True
                    #print("Take no action")






    def observations(self):

        obs = []

        for agent_idx, acid in enumerate(bs.traf.id):
            h = []
            ac = bs.traf.id2idx(acid)
            h.append(bs.traf.lat[agent_idx]/100)
            h.append(bs.traf.lon[agent_idx]/100)
            h.append(bs.traf.hdg[agent_idx]/180 -1)
            h.append(vcas2mach(bs.traf.cas[agent_idx],bs.traf.alt[agent_idx] * 0.3048))
            obs.append(h)
            #obs.extend(h)#
        if self.ini_n_planes < self.max_agents:
            for i in range(len(bs.traf.id)+1, len(bs.traf.id)+(self.max_agents - self.ini_n_planes)+1):
                h = []
                h.append(0)
                h.append(0)
                h.append(0)
                h.append(0)

                obs.append(h)


        return obs




    def step(self):
        in_loss = 0
        # print("Time is ",bs.sim.simt, "-----------------------------------------------")

        a = copy.deepcopy(self.distinm)
        for i in range(self.timestep):
            self.simstep()
            self.save_close_values2()
            done = self.check_episode_done()
            # self.update_agents()
            # print("     2.Time is ",bs.sim.simt, "-----------------------------------------------")
            if done:
                print("EPISODE IS DONE")
                break
        self.n_conflicts = self.update_n_conflicts()
        self.save_close_values2()
        self.update_agents()
        done = self.check_episode_done()
        states = self.observations()
        adj = self.adjency_matrix()
        reward = self.reward()
        self.prev_n_conflicts = copy.deepcopy(self.n_conflicts)
        self.prev_distinm = copy.deepcopy(self.distinm)

        self.distinm_diff = self.distinm - a
        if len(bs.traf.cd.lospairs_unique) > 0:
            if self.first_t_loss == -1:
                self.first_t_loss = bs.sim.simt
            in_loss = 1

        return states, reward, adj, done,len(bs.traf.cd.lospairs_all), len(bs.traf.cd.nmacpairs_all), in_loss

    def check_agent_done(self):
        pass #shif a ka arrit goal state boti

    def calculate_new_conflicts(self):
        new_conflicts = np.zeros(shape=(2))
        for acid_idx, acid in enumerate(list(self.confpairs.keys())[0]):
            ac = bs.traf.id.index(acid)
            new_conflicts[acid_idx] = max(0,(self.n_conflicts[ac]-self.prev_n_conflicts[ac]))
        return new_conflicts

    def save_close_values2(self):
        for acid in bs.traf.id:
            ac = bs.traf.id.index(acid)
            length = len(self.logger.dataloggers[acid].lat)
            qdr, distinm = bs.tools.geo.qdrdist(self.logger.dataloggers[acid].lat, self.logger.dataloggers[acid].lon,
                                                np.ones(length) * bs.traf.lat[ac], np.ones(length) * bs.traf.lon[ac])
            distinm = np.nan_to_num(distinm)
            minindex = np.where(distinm == min(distinm))

            minindex = minindex[0][0]
            distinm = distinm[minindex]
            qdr = qdr[minindex]
            self.save_close_values(ac, minindex, distinm, qdr)
            # print("1", bs.traf.cd.inconf[idx], acid, bs.traf.cd.confpairs_unique)
            if bs.traf.cd.inconf[ac] and acid not in self.agents_step:
                self.agents_step.append(acid)

    def conflict_solved(self):
        confsolved = np.zeros(2)
        for confpair_idx, confpair in enumerate(list(self.confpairs.keys())[0]):
            if confpair not in bs.traf.cd.confpairs_unique:
                confsolved[confpair_idx] = 1
        return confsolved

    def reset(self):
        conflicts = False
        if self.scenario.traffic is not None:
            self.scenario.restore()
        while conflicts == False:
            bs.sim.reset()
            generated_hdgs = loadscn()
            self.logger.reset(bs.traf.id)
            self.scenario.save_scenario()
            self.agents_id_idx = {bs.traf.id[i]: i for i in range(bs.traf.ntraf)}
            self.agents_id = bs.traf.id
            f_conf = -1
            t_loss = 0
            done = False
            all_confs = []
            all_nmacs = []
            while bs.sim.simt <= self.simulation_time:
                all_confs.append(len(bs.traf.cd.lospairs_all))
                all_nmacs.append(len(bs.traf.cd.nmacpairs_all))
                self.logger.logg(bs.traf.lat, bs.traf.lon, bs.traf.alt, bs.traf.cas, bs.traf.hdg, bs.traf.id)
                if len(bs.traf.cd.lospairs_unique) > 0:
                    if f_conf == -1:
                        f_conf = bs.sim.simt
                    t_loss += 1
                if len(bs.traf.cd.confpairs_unique) > 0:
                    conflicts = True

                self.simstep()
        no_conf = max(all_confs)
        no_nmacs = max(all_confs)



        self.confpairs = {}
        self.conf_ac = []
        self.confpair = None
        bs.sim.reset()
        self.scenario.load_scenario()

        while len(bs.traf.cd.confpairs_unique) == 0:
            self.simstep()
            if bs.sim.simt > self.simulation_time:
                done = True
                break

        #try:
        #    self.confpairs[list(bs.traf.cd.confpairs_unique)[0]] = [False, False]
        #    self.confpair = list(bs.traf.cd.confpairs_unique)[0]
        #except:
        #    self.reset()

        self.tcpa = bs.traf.cd.tcpa[0]
        self.agents_step = []
        self.confpairs_list = []
        self.prev_n_conflicts = np.zeros(bs.traf.ntraf)
        self.prev_distinm = np.zeros(bs.traf.ntraf)
        self.n_conflicts = self.update_n_conflicts()
        self.update_agents()
        #states = self.calculate_states()
        states = self.observations()
        #state = self.calculate_general_state()
        indexes = [self.agents_id_idx[acid] for acid in self.agents_step]
        # self.change_ffmode(mode=False)

        for acid in bs.traf.id:
            ac = bs.traf.id.index(acid)
            lat = bs.traf.lat[ac]
            lon = bs.traf.lon[ac]
            lat2 = self.logger.dataloggers[acid].lat[-1]
            lon2 = self.logger.dataloggers[acid].lon[-1]
            stack.stack("LINE " + acid + " " + str(lat) + " " + str(lon) + " " + str(lat2) + " " + str(lon2))
        return states, done,no_conf,no_nmacs, t_loss, f_conf,generated_hdgs

    def check_episode_done(self):

        if bs.sim.simt > self.simulation_time:
            print("Conflicts not solved in time")
            return True

        elif len(self.conf_ac) == 0 and self.t_last_solved >= 5:
            print("No more conflicts present")
            return True

        else:
            print("asnjeri kush nuk plotsohet")
            return False

    def adjency_matrix(self):

        adj = np.zeros((self.max_agents, self.max_agents))


        for idx, agent in enumerate(bs.traf.id):
            for pair in bs.traf.cd.confpairs_unique:
                lpair = list(pair)
                if agent in lpair:
                    idx1 = lpair.index(agent)
                    if idx1 == 0:
                        adj[idx][bs.traf.id2idx(lpair[1])] = 1
                    elif idx1 == 1:
                        adj[idx][bs.traf.id2idx(lpair[0])] = 1
        return adj






    def simstep(self):
        bs.sim.step()
        bs.net.step()

    def check_stats(self):
        lospairs = {}
        for agent_idx, acid in enumerate(list(self.confpairs.keys())[0]):
            ac = bs.traf.id.index(acid)
            lospairs[agent_idx] = [0, agent_idx]
            for idx, confpair in enumerate(bs.traf.cd.lospairs):
                if confpair[0] == acid:
                    lospairs[agent_idx][0] += 1
        return lospairs

    def check_final_stats(self, ac):
        acid = bs.traf.id[ac]
        lat_final = self.logger.dataloggers[acid].lat[-1]
        lon_final = self.logger.dataloggers[acid].lon[-1]
        lat = bs.traf.lat[ac]
        lon = bs.traf.lon[ac]
        dist_final = bs.tools.geo.latlondist(lat, lon, lat_final, lon_final)
        return dist_final

    def change_ffmode(self, mode=True):
        bs.sim.ffmode = mode
        bs.sim.dtmult = 5.0

    def update_n_conflicts(self):
        n_conflicts = np.zeros(bs.traf.ntraf)
        for acid in bs.traf.id:
            ac = bs.traf.id.index(acid)
            for confpair in bs.traf.cd.confpairs_unique:
                for confmember in confpair:
                    if confmember == acid:
                        n_conflicts[ac] += 1
        return n_conflicts


class Saved_scenario():
    def __init__(self):
        self.traffic = None
        self.simulation = None
        self.screen = None
        self.net = None
        self.old_traf = None
        self.old_sim = None
        self.old_scr = None
        self.old_net = None

    def save_scenario(self):
        self.traffic = copy.deepcopy(bs.traf)
        self.simulation = copy.deepcopy(bs.sim)
        self.screen = copy.deepcopy(bs.scr)
        self.net = copy.deepcopy(bs.net)

    def load_scenario(self):
        self.old_traf = bs.traf
        self.old_sim = bs.sim
        self.old_scr = bs.scr
        self.old_net = bs.net
        bs.traf = self.traffic
        bs.sim = self.simulation
        bs.scr = self.screen
        bs.net = self.net

    def restore(self):
        bs.traf = self.old_traf
        bs.sim = self.old_sim
        bs.scr = self.old_scr
        bs.net = self.old_net
