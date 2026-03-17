import sys
sys.path.insert(0, '')
from env import Env

env = Env('dataSet/data.xml')
obs, v, m = env.reset()
print('OK:', obs.keys())
print('Match score:', obs.get('match_score'))
