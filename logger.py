import numpy as np


class DataLoggerLists:
    def __init__(self, acids):
        self.dataloggers = {}
        for acid in acids:
            self.dataloggers[acid] = DataLogger()

    def logg(self, lats, lons, alts, cass, hdg, acids):
        for i, lat in enumerate(lats):
            self.dataloggers[acids[i]].logg(lats[i],lons[i],alts[i],cass[i],hdg[i])

    def reset(self, acids):
        self.dataloggers = {}
        for acid in acids:
            self.dataloggers[acid] = DataLogger()
    
class DataLogger:

    def __init__(self):
        self.lat = np.array([])
        self.lon = np.array([])
        self.alt = np.array([])
        self.cas = np.array([])
        self.hdg = np.array([])

    def logg(self, lat, lon, alt, cas, hdg):
        self.lat = np.append(self.lat, lat)
        self.lon = np.append(self.lon, lon)
        self.alt = np.append(self.alt, alt)
        self.cas = np.append(self.cas, cas)
        self.hdg = np.append(self.hdg, hdg)

    def reset(self):
        self.lat = np.array([])
        self.lon = np.array([])
        self.alt = np.array([])
        self.cas = np.array([])
        self.hdg = np.array([])
