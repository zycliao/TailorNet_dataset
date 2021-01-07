## usage
1. style_shape.py
prepare for simulation
2. Cloth simulation in Marvelous Designer
```
avatar: ROOT/style_shape/up_beta{}.obj
garment: ROOT/style_shape/input_beta{}_gamma{}.obj
motion: ROOT/style_shape/motion_{}.pc2

export: ROOT/style_shape/beta{}_gamma{}.obj
```  

3. vis_simulation.py
visualize simulation results
4. choose_avail.py
select all bad results (e.g., too big shirt on a small body)
and run choose_avail.py
5. post_proc.py