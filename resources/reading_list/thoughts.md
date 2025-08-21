## Modeling and Rendering Non-Euclidean Spaces approximated with Concatenated Polytopes
M-dimensional spaces and manifolds are rendered and traversed by building an approximate scene using polytopes.
In theory, rays are marched over the curved surface of the manifold until it hits geometric primitive.
However, solving for intersections is too expensive and rays have to be projected onto the surface continuously.

Thus, the manifold is discretized with concatenated polytopes and rays can be marched over the flat surfaces.
Lighting and shadows are done with tracking the visited polytopes of the camera ray. On intersection, we
compute lighting by flattening the visited polytopes out and casting a shadow ray from the point to the light.

Scenes are modeled using a mass-spring model to minimize distance between connected polytopes (really cool).

Questions:
Is it possible to traverse implicitly around a curved space without discretized geometries?
If we compute geodesic distance, can we find the resulting point of the curved ray?
CSG for non-euclidean space? Good for modelling and level design

## Higher Dimensional Graphics: Conceiving Worlds in Four Spatial Dimensions and Beyond
The names for the two 4D views are cross-section and frustrum projection. Good to know!

## Rendering Curved Spacetime in Everyday Scenes
Ray paths are curved around a geodesic through an RK4 integration scheme with the einstein field equations

Questions:
It seems like the naive implementation can be optimized quite a bit? 
Doesn't support arbitrary geodesics (We can define curved primitives in non-euclidean space? Which can bend light as well?)

## Real-Time GPU Rendering of Piecewise Algebraic Surfaces
Only supports 4D space at the moment. Uses smooth bezier tetrahedra as primitive

## Shadow-Driven 4D Haptic Visualization
Manipulate n-dimensional objects by frustrum projecting into n-1-dimensions. Then you can manipulate the projection laterally over the n-1-hyperplane
Or have an extra slider for moving the nth coordinate.

Pretty interesting, but they use discretized geometries and pinch points. Can we have a continuous manipulation method with smooth sdf primitives?

## Hyper-Dimensional Deformation Simulation
*Meshing can use higher dimensions to remove self intersections 

## Polyvision: 4D Space Manipulation through Multiple Projections
Similar to shadow-driven haptic visualization but with 4 fixed projections being manipulated simultaneously.
*We could build a system of manipulation within arbitrary number of dynamic views by combining the shadow visualization with multiple cameras.

## Visualization of Nil, Sol, and SL_2(R) geometries
Rays are marched around a geodesic by numerical integration for special reimannian manifolds (M, g)
Only local illumination is considered
"Every compact three-dimensional manifold decomposes into pieces with the geometry modeled by Thurston"
Perhaps we can use these as neural graphics primitives to fit a non euclidean scene? :O
^ only works on 3 manifolds though...

Algorithmically discovering 4D thurston geometry?

