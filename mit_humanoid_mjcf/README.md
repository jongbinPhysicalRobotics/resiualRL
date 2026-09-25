# MIT Humanoid MJCF (converted)

Source URDF: hojae-io/LearningHumanoidArmMotion-RAL2025-Code
  resources/mit_humanoid/urdf/humanoid_full_sf.urdf (commit d176a14)
  - same file name referenced by se-hwan/cusadi (branch wip) cusadi/controllers/mit_humanoid_mpc.py

Conversion: MuJoCo 3.14 URDF importer (fusestatic=false, balanceinertia=true, visual meshes kept)
Added by hand: freejoint "floating_base", 18 torque motors (ctrlrange = URDF effort limit),
  timestep 1 ms / implicitfast, floor scene (scene.xml).
NOT from source (unknown, set yourself): joint damping/armature/frictionloss, contact solref/solimp.

18 actuated joints: a01-a05 right leg, a06-a10 left leg, a11-a14 right arm, a15-a18 left arm
Mass 24.89 kg. Collision: box torso, cylinder shanks/thighs, single cylinder per foot (line contact, "sf").
