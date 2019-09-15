import json
class RoboConfig():
    def __init__(self,cfg_file = "./config.cfg"):
        self.cfg_file = cfg_file
    def load(self):
        with open(self.cfg_file,"r") as f:
            cfg = f.read()
            if cfg:
                self.cfg = json.loads(cfg)
    def dump(self):
        with open(self.cfg_file,"w") as f:
            f.write(json.dumps(self.cfg))
    def reset(self):
        self.cfg = {
            "cap":0,
            "isVideo":False,
            "isRecord":False,
            "recordPath":"./record",
            "red_config":{
                "hmin":4,
                "hmax":17,
                "smin":203,
                "smax":360,
                "vmin":27,
                "vmax":258,
                "thre1":5,
                "thre2":11,
                "ratio_min":1,
                "ratio_max":9,
                "offset_x":150,
                "offset_y":0
            },
            "blue_config":{
                "hmin":111,
                "hmax":180,
                "smin":228,
                "smax":360,
                "vmin":22,
                "vmax":360,
                "thre1":5,
                "thre2":27,
                "ratio_min":1,
                "ratio_max":9,
                "offset_x":150,
                "offset_y":0
            }

        }
    def getcfg(self):
        return self.cfg
    def setColorConfig(self,mode = "red",color_cfg = None):
        if not color_cfg:
            return None
        self.cfg[mode + "_config"] = color_cfg
