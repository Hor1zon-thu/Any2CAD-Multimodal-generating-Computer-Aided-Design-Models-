import cadquery as cq
# Generating a workplane for sketch 0
wp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(0.4765625, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop0=wp_sketch0.moveTo(0.0, 0.0).threePointArc((0.07421875, -0.07421875), (0.1484375, 0.0)).lineTo(0.0, 0.0).close()
solid0=wp_sketch0.add(loop0).extrude(0.296875)
solid=solid0
# Generating a workplane for sketch 1
wp_sketch1 = cq.Workplane(cq.Plane(cq.Vector(-0.4453125, -0.296875, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))
loop1=wp_sketch1.moveTo(1.1953125, 0.0).lineTo(1.1953125, 0.6039473684210526).lineTo(0.0, 0.6039473684210526).lineTo(0.0, 0.4529605263157894).lineTo(0.4026315789473684, 0.4529605263157894).lineTo(0.4026315789473684, 0.15098684210526314).lineTo(0.0, 0.15098684210526314).lineTo(0.0, 0.0).close()
loop2=wp_sketch1.moveTo(0.9939967105263158, 0.3019736842105263).circle(0.07549342105263157)
solid1=wp_sketch1.add(loop1).add(loop2).extrude(0.296875)
solid=solid.union(solid1)
cq.exporters.export(solid,'box2.step')