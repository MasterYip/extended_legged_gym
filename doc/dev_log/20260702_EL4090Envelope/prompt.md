
## Envelope Analyzer for EL4090 Spider Robot

For @extended_legged_gym/legged_gym/legged_gym/envs/el_4090/spider_nomal/el_4090.py , I hope you implement a envelope analyzer including:
1. envelope visualize (the envelope of the robot visualization in gym
2. envelope calculator: The envelope definition is a hexagonal prism. 2d hexagon envelop and robot max height. For now, please use a type of body position (like body with name containing FOOT) Height can be readed from base height (with some bias)

Impl details:
1. Implement a envelope util class in  extended_legged_gym/legged_gym/legged_gym/utils/envelope
2. Use this class in a new class el4090_envelope.py in extended_legged_gym/legged_gym/legged_gym/envs/el_4090/envelope derived from  @extended_legged_gym/legged_gym/legged_gym/envs/el_4090/spider_nomal/el_4090.py ,  This new class has el4090_envelope_spider_condig.py, and will visualize the envelope using extended_legged_gym/legged_gym/legged_gym/utils/gym_visualizer.py during training